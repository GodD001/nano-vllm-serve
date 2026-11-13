"""OpenAI-compatible HTTP API (FastAPI) over AsyncLLMEngine.

Streaming contract (matches what `vllm bench serve` and the `openai` client expect):
* one SSE chunk per generated token, even when the token releases no text yet
  (e.g. half of a multi-byte character), so clients that time every chunk see every token;
* `finish_reason` rides on the chunk of the last token (no extra empty chunk);
* with `stream_options.include_usage`, every chunk carries `"usage": null` and a final
  chunk with `"choices": []` carries the usage; then `data: [DONE]`.
"""
import asyncio
import contextlib
import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from nanovllm.engine.async_llm_engine import AsyncLLMEngine, EngineDeadError, QueueFullError, RequestFailedError
from nanovllm.entrypoints.protocol import (ChatCompletionRequest, CompletionRequest, DetokenizeRequest,
                                           TokenizeRequest)
from nanovllm.sampling_params import SamplingParams

_DISCONNECT_POLL_S = 0.5


class _AbortOnExitStreamingResponse(StreamingResponse):
    """Calls on_exit() however the response ends. Starlette always calls the response once the
    endpoint returns it, even if the client is already gone and the body generator is never
    started; a generator's own `finally` would not run in that case."""

    def __init__(self, content, on_exit, **kwargs):
        super().__init__(content, **kwargs)
        self._on_exit = on_exit

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_exit()


class APIError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error"):
        self.status, self.message, self.err_type = status, message, err_type


def _error(status: int, message: str, err_type: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": err_type, "param": None, "code": status}},
                        status_code=status)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


_json_str = json.JSONEncoder(ensure_ascii=False).encode   # string -> JSON string literal (C fast path)


def _json_reason(reason: str | None) -> str:
    return "null" if reason is None else '"' + reason + '"'


def create_app(engine: AsyncLLMEngine, served_model_name: str, model_path: str | None = None,
               manage_engine: bool = True) -> FastAPI:
    """manage_engine: start/shutdown the engine in the app lifespan (False in tests that
    drive the engine lifecycle themselves)."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if manage_engine:
            await engine.start()
        try:
            yield
        finally:
            if manage_engine:
                await engine.shutdown()

    app = FastAPI(title="nano-vllm-serve", lifespan=lifespan)
    created = int(time.time())
    model_names = {served_model_name} | ({model_path} if model_path else set())

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        return _error(exc.status, exc.message, exc.err_type)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return _error(400, str(exc.errors()), "invalid_request_error")

    # ------------------------------------------------------------------ helpers
    def check_model(name: str | None):
        if name is not None and name not in model_names:
            raise APIError(404, f"model '{name}' not found; served: '{served_model_name}'", "not_found_error")

    def sampling_params(req, max_tokens: int, prompt_len: int) -> tuple[SamplingParams, list[str]]:
        if req.n != 1:
            raise APIError(400, "only n=1 is supported")
        if getattr(req, "echo", False):
            raise APIError(400, "echo is not supported")
        if req.logprobs is not None and req.logprobs is not False:   # identity: 0 == False would pass
            raise APIError(400, "logprobs are not supported")
        stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        top_k = req.top_k if req.top_k not in (None, 0) else -1
        try:
            sp = SamplingParams(
                temperature=1.0 if req.temperature is None else req.temperature,
                max_tokens=max_tokens,
                ignore_eos=req.ignore_eos,
                top_k=top_k,
                top_p=1.0 if req.top_p is None else req.top_p,
                stop_token_ids=list(req.stop_token_ids or []),
            )
        except ValueError as e:
            raise APIError(400, str(e)) from None
        return sp, stop

    def tokenize_prompt(prompt) -> list[int]:
        if isinstance(prompt, str):
            return engine.tokenizer.encode(prompt)
        if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
            return list(prompt)
        if isinstance(prompt, list) and len(prompt) == 1:
            return tokenize_prompt(prompt[0])
        raise APIError(400, "prompt must be a string or a list of token ids (batched prompts are not supported)")

    def chat_prompt(req: ChatCompletionRequest) -> list[int]:
        messages = []
        for m in req.messages:
            content = m.content
            if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
                parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                if len(parts) != len(content):
                    raise APIError(400, "only text content parts are supported")
                content = "".join(parts)
            messages.append({"role": m.role, "content": content or ""})
        # Only template variables (e.g. enable_thinking) may be passed through; keys that would
        # collide with the fixed arguments or change the return type are dropped.
        reserved = {"conversation", "messages", "tokenize", "add_generation_prompt", "return_dict",
                    "return_tensors", "tokenizer_kwargs", "chat_template"}
        kwargs = {k: v for k, v in (req.chat_template_kwargs or {}).items() if k not in reserved}
        try:
            text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except Exception as e:  # noqa: BLE001 - malformed messages / template variables
            raise APIError(400, f"chat template failed: {e}") from None
        return engine.tokenizer.encode(text, add_special_tokens=False)

    def submit(prompt_ids: list[int], sp: SamplingParams, stop: list[str]) -> str:
        try:
            return engine.add_request(prompt_ids, sp, stop)
        except QueueFullError as e:
            raise APIError(429, str(e), "rate_limit_exceeded") from None
        except EngineDeadError as e:
            raise APIError(503, str(e), "service_unavailable") from None
        except ValueError as e:
            raise APIError(400, str(e)) from None

    async def collect(request: Request, rid: str):
        """Non-streaming: gather the whole output; abort if the client disconnects."""
        async def run():
            text, n, reason, n_prompt = [], 0, None, 0
            async with contextlib.aclosing(engine.stream(rid)) as stream:
                async for out in stream:
                    text.append(out.text)
                    n, reason, n_prompt = out.num_completion_tokens, out.finish_reason, out.num_prompt_tokens
            return "".join(text), n, reason, n_prompt

        task = asyncio.ensure_future(run())
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=_DISCONNECT_POLL_S)
                if done:
                    return task.result()
                if await request.is_disconnected():
                    task.cancel()
                    raise APIError(499, "client disconnected")
        except EngineDeadError as e:
            raise APIError(503, str(e), "service_unavailable") from None
        except RequestFailedError as e:
            raise APIError(500, str(e), "server_error") from None
        finally:
            if not task.done():
                task.cancel()
            engine.abort(rid)   # no-op if consumed; covers a task cancelled before it started

    def usage(n_prompt: int, n_completion: int) -> dict:
        return {"prompt_tokens": n_prompt, "completion_tokens": n_completion, "total_tokens": n_prompt + n_completion}

    async def sse(rid: str, make_choice, include_usage: bool, base: dict, first_chunk: dict | None = None) -> AsyncIterator[str]:
        """make_choice(out) returns the choice object already serialized as JSON. The fixed
        part of every chunk (id, object, created, model, usage) is serialized once here, so
        per token only the new text is JSON-escaped (json.dumps per chunk showed up in the
        event loop's profile)."""
        n_prompt = n_completion = 0
        head = "data: " + _dumps(base)[:-1] + ',"choices":['
        tail = ("]," + '"usage":null}\n\n') if include_usage else "]}\n\n"
        try:
            try:
                if first_chunk is not None:
                    yield f"data: {_dumps(first_chunk)}\n\n"
                async with contextlib.aclosing(engine.stream(rid)) as stream:
                    async for out in stream:
                        n_prompt, n_completion = out.num_prompt_tokens, out.num_completion_tokens
                        yield head + make_choice(out) + tail
                if include_usage:
                    yield f"data: {_dumps({**base, 'choices': [], 'usage': usage(n_prompt, n_completion)})}\n\n"
            except RequestFailedError as e:
                code = 503 if isinstance(e, EngineDeadError) else 500
                yield f"data: {_dumps({'error': {'message': str(e), 'type': 'server_error', 'code': code}})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Covers a disconnect at any point, including before engine.stream() was entered
            # (e.g. right after the first chat chunk). No-op once the request was consumed.
            engine.abort(rid)

    def streaming_response(rid: str, body: AsyncIterator[str]) -> StreamingResponse:
        return _AbortOnExitStreamingResponse(body, on_exit=lambda: engine.abort(rid), media_type="text/event-stream")

    # ------------------------------------------------------------------ routes
    @app.get("/health")
    async def health():
        if engine.is_healthy:
            return Response(status_code=200)
        return Response(status_code=503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{
            "id": served_model_name, "object": "model", "created": created, "owned_by": "nano-vllm-serve",
            "root": model_path or served_model_name, "parent": None, "max_model_len": engine.max_model_len,
        }]}

    @app.get("/metrics")
    async def metrics():
        engine.metrics.update_snapshot(engine.snapshot)
        return Response(engine.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/tokenize")
    async def tokenize(req: TokenizeRequest):
        check_model(req.model)
        tokens = engine.tokenizer.encode(req.prompt, add_special_tokens=req.add_special_tokens)
        return {"count": len(tokens), "max_model_len": engine.max_model_len, "tokens": tokens}

    @app.post("/detokenize")
    async def detokenize(req: DetokenizeRequest):
        check_model(req.model)
        return {"prompt": engine.tokenizer.decode(req.tokens)}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        check_model(req.model)
        prompt_ids = tokenize_prompt(req.prompt)
        max_tokens = req.max_tokens if req.max_tokens is not None else 16
        sp, stop = sampling_params(req, max_tokens, len(prompt_ids))
        rid = submit(prompt_ids, sp, stop)
        base = {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()),
                "model": served_model_name}
        if req.stream:
            def make_choice(out):
                return ('{"index":0,"text":' + _json_str(out.text) + ',"logprobs":null,"finish_reason":'
                        + _json_reason(out.finish_reason) + "}")
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            return streaming_response(rid, sse(rid, make_choice, include_usage, base))
        text, n, reason, n_prompt = await collect(request, rid)
        return {**base, "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": reason}],
                "usage": usage(n_prompt, n)}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        check_model(req.model)
        prompt_ids = chat_prompt(req)
        max_tokens = next((v for v in (req.max_completion_tokens, req.max_tokens) if v is not None),
                          engine.max_model_len - len(prompt_ids))
        if max_tokens < 1:
            raise APIError(400, "max_tokens must be >= 1 and the prompt must fit in max_model_len")
        sp, stop = sampling_params(req, max_tokens, len(prompt_ids))
        rid = submit(prompt_ids, sp, stop)
        base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": served_model_name}
        if req.stream:
            def make_choice(out):
                delta = '{"content":' + _json_str(out.text) + "}" if out.token_ids or out.text else "{}"
                return '{"index":0,"delta":' + delta + ',"logprobs":null,"finish_reason":' + _json_reason(out.finish_reason) + "}"
            first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                          "logprobs": None, "finish_reason": None}]}
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            if include_usage:
                first["usage"] = None
            return streaming_response(rid, sse(rid, make_choice, include_usage, base, first))
        text, n, reason, n_prompt = await collect(request, rid)
        return {**base, "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "logprobs": None, "finish_reason": reason}],
                "usage": usage(n_prompt, n)}

    return app
