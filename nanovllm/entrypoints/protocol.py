"""OpenAI-compatible request schemas. Unknown fields are ignored (extra="ignore"), so
clients that send vendor-specific fields (e.g. `repetition_penalty: 1.0`) still work.

Accepted but ignored: `seed` (sampling is not seeded per request). Rejected with 400:
`n != 1`, `echo`, `logprobs`, batched prompts."""
from typing import Any

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StreamOptions(_Model):
    include_usage: bool = False


class _SamplingFields(_Model):
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = 1
    logprobs: Any = None
    seed: int | None = None


class CompletionRequest(_SamplingFields):
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int | None = 16      # OpenAI default for /v1/completions
    echo: bool = False


class ChatMessage(_Model):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None


class TokenizeRequest(_Model):
    model: str | None = None
    prompt: str
    add_special_tokens: bool = True


class DetokenizeRequest(_Model):
    model: str | None = None
    tokens: list[int]
