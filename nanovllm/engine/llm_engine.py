import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


@dataclass(slots=True)
class TokenOutput:
    """One new token for one sequence, produced by one engine step."""
    seq_id: int
    token_id: int
    finished: bool
    finish_reason: str | None  # "stop" | "length" | None


@dataclass(slots=True)
class StepStats:
    """What the last step did (for metrics / progress bars)."""
    kind: str                 # "prefill" | "decode" | "mixed" (decode_first only)
    num_seqs: int
    num_tokens: int           # tokens run through the model this step
    start_time: float         # perf_counter() before scheduling
    end_time: float           # perf_counter() after postprocess
    newly_scheduled: list[int]  # seq_ids scheduled for the first time this step


class LLMEngine:

    def __init__(self, model, **kwargs):
        # Imported lazily so the scheduler / engine logic can be imported and tested
        # on machines without the GPU stack (torch, triton, flash-attn).
        import torch.multiprocessing as mp
        from nanovllm.engine.model_runner import ModelRunner

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.last_step: StepStats | None = None
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def abort(self, seq_id: int) -> Sequence | None:
        return self.scheduler.abort(seq_id)

    def step(self) -> list[TokenOutput]:
        """Run one scheduling step. Returns one TokenOutput per sequence that received a
        new token this step; details of the step are left in `self.last_step`."""
        start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        newly_scheduled = []
        for seq in seqs:
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = start
                newly_scheduled.append(seq.seq_id)
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        if not is_prefill:
            kind = "decode"      # decode-only: upstream decode path (CUDA graphs)
        elif any(not seq.is_prefill for seq in seqs):
            kind = "mixed"       # decode_first: decodes + prefill chunks, one varlen forward
        else:
            kind = "prefill"
        # A mixed step runs through the prefill (varlen) path: each decode sequence is a
        # 1-token chunk whose keys are its cached prefix, read through its block table.
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        produced = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self.last_step = StepStats(
            kind=kind,
            num_seqs=len(seqs),
            num_tokens=num_tokens,
            start_time=start,
            end_time=perf_counter(),
            newly_scheduled=newly_scheduled,
        )
        return [TokenOutput(seq.seq_id, token_id, seq.is_finished, seq.finish_reason) for seq, token_id in produced]

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seqs = {}
        for prompt, sp in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sp)
            seqs[seq.seq_id] = seq
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            step_outputs = self.step()
            stats = self.last_step
            elapsed = stats.end_time - stats.start_time
            if stats.kind == "prefill":
                prefill_throughput = stats.num_tokens / elapsed
            else:
                decode_throughput = stats.num_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for out in step_outputs:
                if out.finished:
                    outputs[out.seq_id] = seqs[out.seq_id].completion_token_ids
                    pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
