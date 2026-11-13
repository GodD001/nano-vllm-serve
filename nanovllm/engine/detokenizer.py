"""Incremental detokenization for streaming.

Each request keeps a small sliding window of token ids. For every new token we decode
  prefix = ids[prefix_offset:read_offset]   and   full = ids[prefix_offset:]
and emit full[len(prefix):] once it is longer than the prefix and does not end in U+FFFD
(an incomplete multi-byte UTF-8 character). The window then slides forward, so each token
costs O(window) and a request costs O(n), not O(n^2). Same idea as vLLM / TGI.

All requests that received a token in one engine step are decoded with a single
`tokenizer.batch_decode` call (`detokenize_step`), so the per-token Python overhead stays
small and most of the work happens inside the tokenizer's native code.
"""

# How many prompt tokens seed the window, so the first generated token is decoded with
# context (e.g. leading-space handling in byte-level BPE).
_PROMPT_CONTEXT = 5
# Drop ids before the window once this many have accumulated.
_TRIM_AT = 64


class IncrementalDetokenizer:

    def __init__(self, prompt_token_ids: list[int], stop: list[str] | None = None):
        self.ids = list(prompt_token_ids[-(_PROMPT_CONTEXT + 1):])
        self.read_offset = len(self.ids)
        self.prefix_offset = max(self.read_offset - _PROMPT_CONTEXT, 0)
        self.stop = [s for s in (stop or []) if s]
        self.max_stop_len = max((len(s) for s in self.stop), default=0)
        self.pending = ""        # decoded but not yet emitted (held back for stop matching)
        self.chunks: list[str] = []
        self.stopped = False     # a stop string was found; caller should finish the request

    # -- per-step protocol (see detokenize_step) --------------------------------------
    def append(self, token_id: int):
        self.ids.append(token_id)

    def windows(self) -> tuple[list[int], list[int]]:
        return self.ids[self.prefix_offset:self.read_offset], self.ids[self.prefix_offset:]

    def update(self, prefix_text: str, full_text: str) -> str:
        """Feed the decoded windows; returns the text that is safe to emit now (may be "")."""
        if len(full_text) > len(prefix_text) and not full_text.endswith("�"):
            self.pending += full_text[len(prefix_text):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            if self.prefix_offset >= _TRIM_AT:
                del self.ids[:self.prefix_offset]
                self.read_offset -= self.prefix_offset
                self.prefix_offset = 0
        return self._release()

    def finish(self) -> str:
        """Flush held-back text when the request ends for a reason other than a stop string.
        An incomplete trailing multi-byte character is dropped, never emitted."""
        out, self.pending = self.pending, ""
        self.chunks.append(out)
        return out

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    # -- internals ----------------------------------------------------------------------
    def _release(self) -> str:
        if self.stopped:
            return ""
        if self.stop:
            hits = [i for s in self.stop if (i := self.pending.find(s)) != -1]
            if hits:
                out = self.pending[:min(hits)]
                self.pending = ""
                self.stopped = True
                self.chunks.append(out)
                return out
            hold = self.max_stop_len - 1
            cut = max(len(self.pending) - hold, 0)
            out, self.pending = self.pending[:cut], self.pending[cut:]
        else:
            out, self.pending = self.pending, ""
        self.chunks.append(out)
        return out


def batch_decode(tokenizer, seqs: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
    """Decode many short id windows at once.

    Fast path: the Rust `tokenizers` backend of a HF fast tokenizer (`_tokenizer.decode_batch`).
    transformers' `batch_decode` wraps it with per-item Python work (to_py_obj, per-sequence
    decode); profiling the server showed that wrapper as ~1/4 of the event loop's GIL time.
    The backend never applies `clean_up_tokenization_spaces`, which the prefix/full diff needs
    off anyway (" ." -> "." would break prefix stability). Fallback: transformers with
    clean-up disabled explicitly."""
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None and hasattr(backend, "decode_batch"):
        return backend.decode_batch(seqs, skip_special_tokens=skip_special_tokens)
    return tokenizer.batch_decode(seqs, skip_special_tokens=skip_special_tokens,
                                  clean_up_tokenization_spaces=False)


def detokenize_step(tokenizer, items: list[tuple[IncrementalDetokenizer, int]],
                    skip_special_tokens: bool = True) -> list[str]:
    """Append one token to each detokenizer and return each one's emitted text delta.
    Uses one batch_decode call for the whole step."""
    if not items:
        return []
    prefixes, fulls = [], []
    for detok, token_id in items:
        detok.append(token_id)
        prefix, full = detok.windows()
        prefixes.append(prefix)
        fulls.append(full)
    texts = batch_decode(tokenizer, prefixes + fulls, skip_special_tokens)
    n = len(items)
    return [detok.update(texts[i], texts[n + i]) for i, (detok, _) in enumerate(items)]
