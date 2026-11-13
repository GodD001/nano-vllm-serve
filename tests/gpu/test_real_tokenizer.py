"""Incremental detokenization with the real Qwen3 tokenizer (Chinese, emoji, ZWJ sequences).
Needs `transformers` and the model directory; no GPU. Skips otherwise.
Model path: $NANOVLLM_TEST_MODEL or ~/huggingface/Qwen3-0.6B."""
import os

import pytest

from nanovllm.engine.detokenizer import IncrementalDetokenizer, detokenize_step

MODEL = os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B"))
if not os.path.isfile(os.path.join(MODEL, "tokenizer.json")):
    pytest.skip(f"no tokenizer at {MODEL}", allow_module_level=True)
transformers = pytest.importorskip("transformers")
if not hasattr(transformers.AutoTokenizer, "from_pretrained"):
    pytest.skip("transformers is stubbed", allow_module_level=True)

TEXTS = [
    "你好，世界！今天天气很好，我们去公园散步吧。",
    "Emoji test 🎉🚀👍🏽 flags 🇨🇳🇺🇸 family 👨‍👩‍👧‍👦 done.",
    "混合 mixed 文本 with 数字 12345 和标点。",
    "日本語のテキストと한국어 텍스트도 테스트합니다。",
]


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.mark.parametrize("text", TEXTS)
def test_stream_matches_full_decode(tok, text):
    prompt = tok.encode("Prompt: ")
    ids = tok.encode(text)
    d = IncrementalDetokenizer(prompt)
    deltas = [detokenize_step(tok, [(d, t)])[0] for t in ids] + [d.finish()]
    assert "".join(deltas) == tok.decode(ids, skip_special_tokens=True) == text
    assert all("�" not in x for x in deltas)


def test_some_tokens_are_partial_characters(tok):
    """Guard that the test exercises the hard case: some single tokens decode to U+FFFD."""
    ids = tok.encode(TEXTS[1])
    assert any("�" in tok.decode([t]) for t in ids)


def test_stop_string_with_real_tokenizer(tok):
    ids = tok.encode("答案是：42。谢谢！")
    d = IncrementalDetokenizer(tok.encode("Q: "), stop=["谢谢"])
    out = []
    for t in ids:
        out.append(detokenize_step(tok, [(d, t)])[0])
        if d.stopped:
            break
    assert d.stopped and "".join(out) == "答案是：42。"


def test_fast_path_matches_transformers_decode(tok):
    """The Rust decode_batch fast path must equal transformers' batch_decode with
    clean-up disabled, including special tokens and partial characters."""
    from nanovllm.engine.detokenizer import batch_decode
    ids = tok.encode("<think>\n推理 🎉 done</think>\n答案。<|im_end|>")
    windows = [ids[i:i + 6] for i in range(len(ids))] + [ids]
    for skip in (True, False):
        assert batch_decode(tok, windows, skip) == tok.batch_decode(
            windows, skip_special_tokens=skip, clean_up_tokenization_spaces=False)
