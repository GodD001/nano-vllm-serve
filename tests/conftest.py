"""Test bootstrap for CPU-only machines.

The scheduler / engine / serving tests need no GPU. If `transformers` is missing we
register a minimal stand-in *before* `nanovllm` is imported; with the real package
installed nothing is stubbed. GPU-only tests live in tests/gpu and skip without CUDA.
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

# LLMEngine imports torch / ModelRunner lazily (inside __init__), so the engine, scheduler
# and serving layer import fine without the GPU stack; only `transformers` may need a stub.


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_sequence_block_size():
    """Tests shrink `Sequence.block_size` (a class attribute); put it back afterwards."""
    from nanovllm.engine.sequence import Sequence
    saved = Sequence.block_size
    yield
    Sequence.block_size = saved
