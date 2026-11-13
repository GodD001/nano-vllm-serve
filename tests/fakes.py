"""Test doubles for the serving layer: a byte-level tokenizer, a scripted model runner and an
LLMEngine that uses the real Scheduler / step() / generate() with the fake runner."""
import threading
import time
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence

EOS = 256
REPLY = "Hello, 世界! 🎉 nano-vllm-serve. "  # multi-byte chars split across byte tokens


class ByteTokenizer:
    """ids 0..255 are raw UTF-8 bytes; 256 is EOS. Multi-byte characters always span several
    tokens, which is exactly the case incremental detokenization must handle."""
    eos_token_id = EOS

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:
        out, buf = [], bytearray()
        for i in ids:
            if i < 256:
                buf.append(i)
            elif not skip_special_tokens:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                out.append("<eos>")
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def batch_decode(self, seqs, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> list[str]:
        assert clean_up_tokenization_spaces is False, "incremental detokenization must disable clean-up"
        return [self.decode(s, skip_special_tokens) for s in seqs]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if kwargs.get("raise_error"):
            raise ValueError("template error (simulated)")
        assert tokenize is False and add_generation_prompt is True
        text = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")


class FakeRunner:
    """Emits REPLY's bytes in order for every sequence (so the expected text is known),
    independent of batch composition. Optional per-step delay and a gate to pause steps."""

    def __init__(self, reply: str = REPLY, step_delay: float = 0.0, eos_after: int | None = None):
        self.reply = list(reply.encode("utf-8"))
        self.step_delay = step_delay
        self.eos_after = eos_after
        self.gate = threading.Event()
        self.gate.set()
        self.num_steps = 0
        self.exited = False

    def call(self, method, *args):
        if method == "exit":
            self.exited = True
            return None
        assert method == "run"
        seqs, is_prefill = args
        self.gate.wait()
        if self.step_delay:
            time.sleep(self.step_delay)
        self.num_steps += 1
        out = []
        for seq in seqs:
            k = seq.num_completion_tokens
            if self.eos_after is not None and k >= self.eos_after:
                out.append(EOS)
            else:
                out.append(self.reply[k % len(self.reply)])
        return out


class FakeLLMEngine(LLMEngine):

    def __init__(self, runner: FakeRunner | None = None, num_blocks: int = 64, block_size: int = 16,
                 max_num_seqs: int = 16, max_num_batched_tokens: int = 256, max_model_len: int = 512,
                 scheduler_policy: str = "prefill_first", max_prefill_tokens_per_step: int = -1):
        self.config = SimpleNamespace(
            num_kvcache_blocks=num_blocks, kvcache_block_size=block_size, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len, eos=EOS,
            scheduler_policy=scheduler_policy, max_prefill_tokens_per_step=max_prefill_tokens_per_step,
        )
        Sequence.block_size = block_size
        self.model_runner = runner or FakeRunner()
        self.tokenizer = ByteTokenizer()
        self.scheduler = Scheduler(self.config)
        self.ps = []
        self.last_step = None
        self._exited = False
