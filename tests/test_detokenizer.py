"""Incremental detokenization: never emits partial multi-byte characters, handles stop
strings, stays O(n). Uses a byte-level tokenizer (every non-ASCII char spans 2-4 tokens);
tests/gpu/test_real_tokenizer.py repeats the key cases with the Qwen3 tokenizer."""
import pytest

from nanovllm.engine.detokenizer import IncrementalDetokenizer, detokenize_step
from tests.fakes import ByteTokenizer

TEXTS = [
    "Hello, world!",
    "你好，世界！今天天气很好。",
    "emoji 🎉🚀👍🏽 and flags 🇨🇳🇺🇸",
    "mixed 中文 + emoji 😀 + ASCII",
    "家族 👨‍👩‍👧‍👦 zero-width joiner sequence",
]


class CountingTokenizer(ByteTokenizer):
    def __init__(self):
        self.decoded_tokens = 0
        self.calls = 0

    def batch_decode(self, seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        self.calls += 1
        self.decoded_tokens += sum(len(s) for s in seqs)
        return super().batch_decode(seqs, skip_special_tokens, clean_up_tokenization_spaces)


def stream(tok, prompt: str, completion_ids: list[int], stop=None):
    d = IncrementalDetokenizer(tok.encode(prompt), stop)
    deltas = []
    for t in completion_ids:
        deltas.append(detokenize_step(tok, [(d, t)])[0])
        if d.stopped:
            break
    if not d.stopped:
        deltas.append(d.finish())
    return d, deltas


@pytest.mark.parametrize("text", TEXTS)
def test_stream_reassembles_exactly_and_never_emits_partial_chars(text):
    tok = ByteTokenizer()
    d, deltas = stream(tok, "prompt: ", tok.encode(text))
    assert "".join(deltas) == text == d.text
    for delta in deltas:
        assert "�" not in delta
        delta.encode("utf-8")  # every delta is valid UTF-8 on its own


def test_held_back_until_character_completes():
    tok = ByteTokenizer()
    ids = tok.encode("世")  # 3 bytes
    d = IncrementalDetokenizer(tok.encode("x"))
    assert [detokenize_step(tok, [(d, i)])[0] for i in ids] == ["", "", "世"]


def test_incomplete_tail_is_dropped_on_finish():
    tok = ByteTokenizer()
    ids = tok.encode("ok世")[:-1]  # last char cut mid-sequence (e.g. max_tokens hit)
    d, deltas = stream(tok, "p", ids)
    assert "".join(deltas) == "ok"


def test_special_tokens_skipped():
    tok = ByteTokenizer()
    d, deltas = stream(tok, "p", tok.encode("ab") + [256])
    assert "".join(deltas) == "ab"


@pytest.mark.parametrize("stop,expected", [
    (["world"], "Hello, "),
    (["界"], "你好，世"),
    (["🎉"], "party "),
    (["zzz", ", "], "Hello"),
])
def test_stop_strings(stop, expected):
    tok = ByteTokenizer()
    text = {"world": "Hello, world!", "界": "你好，世界！", "🎉": "party 🎉 time", "zzz": "Hello, world!"}[stop[0]]
    d, deltas = stream(tok, "p", tok.encode(text), stop=stop)
    assert d.stopped
    assert "".join(deltas) == expected
    for i, delta in enumerate(deltas):  # no prefix of the stop string leaks out early
        assert not any(s in "".join(deltas[: i + 1]) for s in stop)


def test_stop_string_not_matched_streams_everything():
    tok = ByteTokenizer()
    d, deltas = stream(tok, "p", tok.encode("abc abd abe"), stop=["abf"])
    assert not d.stopped and "".join(deltas) == "abc abd abe"


def test_cost_is_linear_in_length():
    """Total tokens decoded grows linearly (bounded window), not quadratically."""
    costs = {}
    for n_chars in (200, 800):
        tok = CountingTokenizer()
        text = ("中文🎉ab" * n_chars)[:n_chars]
        d, deltas = stream(tok, "p", tok.encode(text))
        assert "".join(deltas) == text
        costs[n_chars] = tok.decoded_tokens / len(tok.encode(text))
    assert costs[800] < costs[200] * 1.2   # per-token cost does not grow with length
    assert costs[800] < 30


def test_one_batch_decode_per_step():
    tok = CountingTokenizer()
    dets = [IncrementalDetokenizer(tok.encode("p")) for _ in range(10)]
    detokenize_step(tok, [(d, ord("a")) for d in dets])
    assert tok.calls == 1
