"""Real uvicorn server on a local port (fake model): official `openai` client, client
disconnects, concurrent HTTP streams, graceful shutdown."""
import asyncio
import threading
import time

import httpx
import pytest
import uvicorn

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.openai_server import create_app
from tests.fakes import REPLY, FakeLLMEngine, FakeRunner

openai = pytest.importorskip("openai")
MODEL = "fake-model"
REPLY_BYTES = REPLY.encode()


def expected_text(n):
    # FakeRunner cycles through REPLY's bytes; a cut multi-byte char at the end is dropped
    return (REPLY_BYTES * (n // len(REPLY_BYTES) + 1))[:n].decode("utf-8", errors="ignore")


class LiveServer:
    def __init__(self, runner=None, **engine_kw):
        self.runner = runner or FakeRunner()
        self.engine = AsyncLLMEngine(lambda: FakeLLMEngine(self.runner, **engine_kw))
        app = create_app(self.engine, served_model_name=MODEL, model_path="/models/fake")
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.01)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
        assert self.runner.exited          # lifespan shut the engine down

    def blocks_free(self):
        s = self.engine.snapshot
        return s.kv_blocks_used == 0 and s.num_running == 0 and s.num_waiting == 0

    def no_leaks(self):
        """Engine idle with all KV blocks free AND no per-request state left on the asyncio side."""
        return self.blocks_free() and not self.engine._requests and self.engine.num_unscheduled == 0


def wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_official_openai_client():
    with LiveServer() as srv:
        client = openai.OpenAI(base_url=srv.url + "/v1", api_key="unused")
        assert [m.id for m in client.models.list()] == [MODEL]

        c = client.completions.create(model=MODEL, prompt="Hello", max_tokens=10)
        assert c.choices[0].text == expected_text(10) and c.choices[0].finish_reason == "length"
        assert c.usage.completion_tokens == 10

        stream = client.completions.create(model=MODEL, prompt="Hello", max_tokens=15, stream=True,
                                           stream_options={"include_usage": True})
        chunks = list(stream)
        text = "".join(ch.choices[0].text for ch in chunks if ch.choices)
        assert text == expected_text(15) and chunks[-1].usage.completion_tokens == 15

        chat = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                              max_tokens=8, temperature=0)
        assert chat.choices[0].message.content == expected_text(8)
        assert chat.choices[0].message.role == "assistant"

        stream = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                                max_tokens=11, stream=True, stream_options={"include_usage": True})
        chunks = list(stream)
        text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
        assert text == expected_text(11)
        assert [ch.choices[0].finish_reason for ch in chunks if ch.choices][-1] == "length"
        assert chunks[-1].usage.completion_tokens == 11
        wait_until(srv.blocks_free)


def test_client_disconnect_frees_blocks():
    with LiveServer(FakeRunner(step_delay=0.005), num_blocks=32) as srv:
        with httpx.Client(timeout=10) as http:
            with http.stream("POST", srv.url + "/v1/completions",
                             json={"prompt": "x", "max_tokens": 400, "stream": True, "ignore_eos": True}) as r:
                lines = r.iter_lines()
                got = [next(lines) for _ in range(6)]
                assert any(line.startswith("data: ") for line in got)
                assert srv.engine.snapshot.kv_blocks_used > 0
            # leaving the `with` closes the connection mid-stream
        wait_until(srv.blocks_free)

        # non-streaming request abandoned by the client
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(srv.url + "/v1/completions", json={"prompt": "y", "max_tokens": 400}, timeout=0.2)
        wait_until(srv.blocks_free)


def test_concurrent_http_streams():
    async def main(url):
        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            async def one(i, n):
                text, count, last = "", 0, None
                async with client.stream("POST", "/v1/completions", json={
                        "prompt": f"req {i}", "max_tokens": n, "stream": True}) as r:
                    assert r.status_code == 200
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        import json
                        ch = json.loads(line[6:])
                        if ch["choices"]:
                            count += 1
                            text += ch["choices"][0]["text"]
                            last = ch["choices"][0]["finish_reason"]
                return n, text, count, last

            results = await asyncio.gather(*[one(i, 5 + (i * 7) % 40) for i in range(20)])
            for n, text, count, last in results:
                assert count == n and text == expected_text(n) and last == "length"

    with LiveServer(max_num_seqs=6, num_blocks=128) as srv:
        asyncio.run(main(srv.url))
        wait_until(srv.blocks_free)


# ---------------------------------------------------------------------------- disconnect / leak
# Long requests (2000 tokens x 5 ms/step = 10 s) with a 3 s deadline: if blocks come back in
# time, it was the abort that freed them, not the request finishing on its own.
LONG = 2000
SLOW = dict(step_delay=0.005)
BIG = dict(num_blocks=512, max_model_len=4096)


@pytest.mark.parametrize("path,payload", [
    ("/v1/completions", {"prompt": "x", "max_tokens": LONG, "stream": True}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": LONG, "stream": True}),
])
def test_disconnect_right_after_headers(path, payload):
    with LiveServer(FakeRunner(**SLOW), **BIG) as srv:
        with httpx.Client(timeout=10) as http:
            with http.stream("POST", srv.url + path, json=payload) as r:
                assert r.status_code == 200      # headers received; body never read
        wait_until(srv.no_leaks, timeout=3)


def test_chat_disconnect_after_first_chunk():
    """The role chunk is sent before engine.stream() is entered (the reviewed leak)."""
    with LiveServer(FakeRunner(**SLOW), **BIG) as srv:
        with httpx.Client(timeout=10) as http:
            with http.stream("POST", srv.url + "/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "x"}], "max_tokens": LONG, "stream": True}) as r:
                first = next(r.iter_lines())
                assert '"role":"assistant"' in first
        wait_until(srv.no_leaks, timeout=3)


def test_many_clients_time_out_no_leaks():
    async def main(url):
        async with httpx.AsyncClient(base_url=url, timeout=0.3) as client:
            async def stream_then_timeout(i):
                async def consume():
                    async with client.stream("POST", "/v1/completions", json={
                            "prompt": f"s{i}", "max_tokens": LONG, "stream": True}) as r:
                        async for _ in r.aiter_lines():
                            pass
                with pytest.raises(asyncio.TimeoutError):   # client-side deadline closes the connection
                    await asyncio.wait_for(consume(), timeout=0.3 + 0.05 * i)
            async def non_stream_timeout(i):
                with pytest.raises(httpx.TimeoutException):
                    await client.post("/v1/completions", json={"prompt": f"n{i}", "max_tokens": LONG})
            await asyncio.gather(*[stream_then_timeout(i) for i in range(8)],
                                 *[non_stream_timeout(i) for i in range(8)])

    with LiveServer(FakeRunner(**SLOW), max_num_seqs=32, **BIG) as srv:
        asyncio.run(main(srv.url))
        wait_until(srv.no_leaks, timeout=4)
