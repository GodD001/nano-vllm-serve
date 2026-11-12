"""Test bootstrap for CPU-only machines.

`nanovllm/__init__.py` imports the full LLM (torch, triton, flash-attn, transformers).
The scheduler / block-manager tests need none of that, so when those packages are
missing we register minimal stand-ins *before* `nanovllm` is imported. On a machine
with the real stack installed nothing is stubbed and the real modules are used.
"""
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None


if _missing("transformers"):
    stub = types.ModuleType("transformers")
    stub.AutoConfig = type("AutoConfig", (), {})
    stub.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = stub

if any(_missing(m) for m in ("torch", "triton", "flash_attn")):
    stub = types.ModuleType("nanovllm.llm")

    class LLM:  # pragma: no cover - placeholder only
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GPU stack (torch/triton/flash-attn) not installed")

    stub.LLM = LLM
    sys.modules["nanovllm.llm"] = stub


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_sequence_block_size():
    """Tests shrink `Sequence.block_size` (a class attribute); put it back afterwards."""
    from nanovllm.engine.sequence import Sequence
    saved = Sequence.block_size
    yield
    Sequence.block_size = saved
