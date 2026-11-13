from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0  # cumulative, for metrics
        self.policy = getattr(config, "scheduler_policy", "prefill_first")
        cap = getattr(config, "max_prefill_tokens_per_step", -1)
        self.max_prefill_tokens_per_step = cap if cap and cap > 0 else self.max_num_batched_tokens

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (seqs, has_prefill). Each seq's num_scheduled_tokens says how many tokens
        it runs this step; a decode sequence has is_prefill == False and 1 token.
        prefill_first steps are all-prefill or all-decode (upstream); decode_first steps may
        mix both (has_prefill and some seqs with is_prefill == False)."""
        if self.policy == "decode_first":
            return self._schedule_decode_first()
        return self._schedule_prefill_first()

    def _schedule_decode_first(self) -> tuple[list[Sequence], bool]:
        # 1. Every running sequence gets one decode token (upstream's append / preempt logic).
        decode_seqs = []
        preempted = False
        while self.running and len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                preempted = True
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
        self.running.extendleft(reversed(decode_seqs))

        # 2. Fill the remaining budget with prefill, FCFS; only the first admitted sequence may
        #    be chunked. Skip admission in a step that had to preempt (KV is exhausted; admitting
        #    would just evict again), unless nothing is decoding (then it is a plain prefill step).
        prefill_seqs = []
        budget = min(self.max_num_batched_tokens - len(decode_seqs), self.max_prefill_tokens_per_step)
        if not preempted or not decode_seqs:
            while self.waiting and budget > 0 and len(decode_seqs) + len(prefill_seqs) < self.max_num_seqs:
                seq = self.waiting[0]
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        break
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                if budget < num_tokens and prefill_seqs:
                    break
                if not seq.block_table:
                    self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_scheduled_tokens = min(num_tokens, budget)
                budget -= seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                prefill_seqs.append(seq)

        seqs = decode_seqs + prefill_seqs
        assert seqs
        return seqs, bool(prefill_seqs)

    def _schedule_prefill_first(self) -> tuple[list[Sequence], bool]:
        # Upstream nano-vLLM policy, unchanged (tests/golden/prefill_first_trace.json pins it).
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.waiting and self.waiting[0].block_table:
            # decode_first only: the head of `waiting` is mid-chunked-prefill and holds KV blocks.
            # Queueing the victim (which holds none) in front of it would block admission while
            # those blocks stay pinned (deadlock once nothing is left running). Under
            # prefill_first a mid-chunk head is always scheduled before any decode, so this
            # branch never runs there and upstream behavior is unchanged.
            self.waiting.insert(1, seq)
        else:
            self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool) -> list[tuple[Sequence, int]]:
        """Apply one step's sampled tokens. Returns (seq, token_id) for every sequence that
        received a new token this step (a partial prefill chunk receives none)."""
        outputs = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Per-sequence: a partial prefill chunk yields no token. (Upstream also checked
            # is_prefill, which is redundant: a decode always completes its sequence.)
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            outputs.append((seq, token_id))
            if (not seq.ignore_eos and token_id == self.eos) or token_id in seq.stop_token_ids:
                seq.finish_reason = "stop"
            elif seq.num_completion_tokens == seq.max_tokens:
                seq.finish_reason = "length"
            if seq.finish_reason is not None:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return outputs

    def abort(self, seq_id: int) -> Sequence | None:
        """Remove a sequence in any state (waiting, mid-chunked-prefill, preempted, running)
        and free its KV blocks. Returns the aborted sequence, or None if it is not live.

        Must not be called while a step is in flight (the engine thread only calls it
        between steps)."""
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    if seq.block_table:
                        self.block_manager.deallocate(seq)
                    seq.num_scheduled_tokens = 0
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_reason = "abort"
                    return seq
        return None
