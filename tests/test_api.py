"""OpenAI API contract (in-process ASGI). Mirrors what `vllm bench serve` and the `openai`
client rely on: one SSE chunk per token, finish_reason on the last token's chunk, usage
chunk with empty choices, [DONE], unknown fields ignored, OpenAI error shapes."""
import asyncio
import json

import httpx
import pytest

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.openai_server import create_app
from tests.fakes import REPLY, FakeLLMEngine, FakeRunner

MODEL = "fake-model"
REPLY_BYTES = REPLY.encode()


def expected_text(n):
    return REPLY_BYTES[:n].decode("utf-8", errors="ignore")


def parse_sse(body: str):
    events = [e for e in body.split("\n\n") if e.strip()]
    assert all(e.startswith("data: ") for e in events), events
    assert events[-1] == "data: [DONE]"
    return [json.loads(e[len("data: "):]) for e in events[:-1]]


def api_test(coro_fn, **engine_kw):
    async def main():
        engine = AsyncLLMEngine(lambda: FakeLLMEngine(engine_kw.pop("runner", None) or FakeRunner(), **engine_kw))
        await engine.start()
        app = create_app(engine, served_model_name=MODEL, model_path="/models/fake", manage_engine=False)
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await coro_fn(client, engine)
        finally:
            await engine.shutdown()
    asyncio.run(main())


def test_health_models_metrics():
    async def t(client, engine):
        assert (await client.get("/health")).status_code == 200
        models = (await client.get("/v1/models")).json()
        assert models["object"] == "list"
        m = models["data"][0]
        assert m["id"] == MODEL and m["root"] == "/models/fake" and m["max_model_len"] == 512
        r = await client.get("/metrics")
        assert r.status_code == 200 and "nanovllm_kv_cache_blocks_total" in r.text
    api_test(t)


def test_completion_non_streaming_and_unknown_fields_ignored():
    async def t(client, engine):
        r = await client.post("/v1/completions", json={
            "model": MODEL, "prompt": "Say hi", "max_tokens": 9,
            "repetition_penalty": 1.0, "some_future_field": {"x": 1}, "logprobs": None})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion" and body["model"] == MODEL
        choice = body["choices"][0]
        assert choice["text"] == expected_text(9) and choice["finish_reason"] == "length"
        assert body["usage"] == {"prompt_tokens": len(b"Say hi"), "completion_tokens": 9,
                                 "total_tokens": len(b"Say hi") + 9}
    api_test(t)


@pytest.mark.parametrize("include_usage", [True, False])
def test_completion_streaming_one_chunk_per_token(include_usage):
    n = 20   # covers multi-byte characters: some tokens release no text
    async def t(client, engine):
        payload = {"model": MODEL, "prompt": "Say hi", "max_tokens": n, "stream": True,
                   "repetition_penalty": 1.0, "logprobs": None, "ignore_eos": True}
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
        r = await client.post("/v1/completions", json=payload)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        chunks = parse_sse(r.text)
        token_chunks = [c for c in chunks if c["choices"]]
        assert len(token_chunks) == n                       # vllm bench serve counts these
        assert "".join(c["choices"][0]["text"] for c in token_chunks) == expected_text(n)
        assert any(c["choices"][0]["text"] == "" for c in token_chunks)   # held-back half char
        assert all(c["choices"][0]["finish_reason"] is None for c in token_chunks[:-1])
        assert token_chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert all("�" not in c["choices"][0]["text"] for c in token_chunks)
        if include_usage:
            assert all(c["usage"] is None for c in token_chunks)
            assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == n
        else:
            assert len(chunks) == n
    api_test(t)


def test_completion_token_id_prompt_and_stop_string():
    async def t(client, engine):
        r = await client.post("/v1/completions", json={"prompt": list(b"abc"), "max_tokens": 50, "stop": ["世界"]})
        body = r.json()
        assert body["choices"][0]["text"] == "Hello, " and body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 3
    api_test(t)


def test_chat_streaming_and_non_streaming():
    async def t(client, engine):
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]}]
        r = await client.post("/v1/chat/completions", json={
            "model": MODEL, "messages": msgs, "max_completion_tokens": 12, "stream": True,
            "stream_options": {"include_usage": True}})
        chunks = parse_sse(r.text)
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        token_chunks = [c for c in chunks[1:] if c["choices"]]
        assert len(token_chunks) == 12
        assert all("content" in c["choices"][0]["delta"] and "role" not in c["choices"][0]["delta"]
                   for c in token_chunks)
        assert "".join(c["choices"][0]["delta"]["content"] for c in token_chunks) == expected_text(12)
        assert token_chunks[-1]["choices"][0]["finish_reason"] == "length"
        usage = chunks[-1]["usage"]
        prompt = "<|system|>be brief\n<|user|>hi there\n<|assistant|>"
        assert usage["prompt_tokens"] == len(prompt.encode()) and usage["completion_tokens"] == 12

        r = await client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 7})
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"] == {"role": "assistant", "content": expected_text(7)}
        assert body["choices"][0]["finish_reason"] == "length"
    api_test(t)


@pytest.mark.parametrize("payload,status", [
    ({"prompt": "x", "n": 2}, 400),
    ({"prompt": "x", "logprobs": 1}, 400),
    ({"prompt": "x", "temperature": -1}, 400),
    ({"prompt": "x", "top_p": 0}, 400),
    ({"prompt": "x", "max_tokens": 10_000}, 400),        # > max_model_len
    ({"prompt": "x", "model": "other-model"}, 404),
    ({"max_tokens": 5}, 400),                            # missing prompt
    ({"prompt": ["a", "b"]}, 400),                       # batched prompts unsupported
])
def test_errors_use_openai_shape(payload, status):
    async def t(client, engine):
        r = await client.post("/v1/completions", json=payload)
        assert r.status_code == status, r.text
        err = r.json()["error"]
        assert err["code"] == status and err["message"] and err["type"]
    api_test(t)


def test_greedy_and_sampling_params_accepted():
    async def t(client, engine):
        for extra in ({"temperature": 0}, {"top_k": 5, "top_p": 0.9, "temperature": 0.7}, {"top_k": -1}):
            r = await client.post("/v1/completions", json={"prompt": "x", "max_tokens": 3, **extra})
            assert r.status_code == 200, (extra, r.text)
    api_test(t)


def test_tokenize_detokenize_roundtrip():
    async def t(client, engine):
        r = await client.post("/tokenize", json={"model": MODEL, "prompt": "héllo", "add_special_tokens": False})
        tokens = r.json()["tokens"]
        assert r.json()["count"] == len("héllo".encode())
        r = await client.post("/detokenize", json={"model": MODEL, "tokens": tokens})
        assert r.json()["prompt"] == "héllo"
    api_test(t)


def test_429_when_waiting_queue_full():
    runner = FakeRunner()
    runner.gate.clear()

    async def t(client, engine):
        engine.max_waiting_requests = 1
        first = asyncio.ensure_future(client.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}))
        await asyncio.sleep(0.05)
        r = await client.post("/v1/completions", json={"prompt": "b", "max_tokens": 2, "stream": True})
        assert r.status_code == 429 and r.json()["error"]["type"] == "rate_limit_exceeded"
        runner.gate.set()
        assert (await first).status_code == 200
    api_test(t, runner=runner)


# ---------------------------------------------------------------------------- disconnect leaks
# Raw ASGI calls whose `send` fails at a chosen message, i.e. the client vanished exactly then.
# Message 0 = http.response.start (body generator never started); 1 = first body chunk (for
# chat: the role chunk, sent before engine.stream() is entered); 2 = second body chunk.

class ClientGone(OSError):
    pass


async def call_with_failing_send(app, path: str, payload: dict, fail_at: int):
    body = json.dumps(payload).encode()
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"content-type", b"application/json"),
                                          (b"content-length", str(len(body)).encode())],
             "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 80)}
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()           # never reports a disconnect on its own

    sent = 0

    async def send(message):
        nonlocal sent
        if sent == fail_at:
            raise ClientGone("client went away")
        sent += 1

    try:
        await app(scope, receive, send)
    except Exception:  # noqa: BLE001 - the server may surface the send error
        pass


@pytest.mark.parametrize("fail_at", [0, 1, 2])
@pytest.mark.parametrize("path,payload", [
    ("/v1/completions", {"prompt": "x", "max_tokens": 400, "stream": True}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 400, "stream": True}),
])
def test_stream_send_failure_never_leaks(path, payload, fail_at):
    async def t(client, engine):
        app = client._transport.app
        await call_with_failing_send(app, path, payload, fail_at)
        deadline = asyncio.get_running_loop().time() + 1
        while engine._requests or engine.snapshot.kv_blocks_used or engine.snapshot.num_running \
                or engine.snapshot.num_waiting:
            assert asyncio.get_running_loop().time() < deadline, (
                f"leak: requests={list(engine._requests)} snapshot={engine.snapshot}")
            await asyncio.sleep(0.01)
    api_test(t, runner=FakeRunner(step_delay=0.005))   # 400 tokens x 5 ms = 2 s: abort must beat it


@pytest.mark.parametrize("path,payload,status", [
    ("/v1/completions", {"prompt": "x", "logprobs": 0}, 400),                 # 0 == False must not pass
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, 400),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 3,
                              "chat_template_kwargs": {"tokenize": True, "add_generation_prompt": False}}, 200),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 3,
                              "chat_template_kwargs": {"raise_error": True}}, 400),
    ("/v1/completions", {"prompt": "x", "max_tokens": 3, "seed": 1234}, 200),  # accepted, ignored (documented)
])
def test_review_api_edge_cases(path, payload, status):
    async def t(client, engine):
        r = await client.post(path, json=payload)
        assert r.status_code == status, r.text
    api_test(t)
