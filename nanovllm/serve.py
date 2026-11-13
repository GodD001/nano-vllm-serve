"""OpenAI-compatible server.

    python -m nanovllm.serve --model ~/huggingface/Qwen3-0.6B --port 8000 --scheduler-policy prefill_first
"""
import argparse
import logging
import os

import uvicorn

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.openai_server import create_app
from nanovllm.metrics import ServerMetrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="nano-vllm-serve: OpenAI-compatible server")
    p.add_argument("--model", required=True, help="local model directory (HF format)")
    p.add_argument("--served-model-name", default=None, help="model id exposed by the API (default: directory name)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scheduler-policy", default="prefill_first", choices=["prefill_first"],
                   help="scheduling policy (decode_first lands in Phase 3)")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-waiting-requests", type=int, default=1024,
                   help="reject new requests with HTTP 429 when this many are waiting to be scheduled")
    p.add_argument("--log-level", default="info")
    p.add_argument("--access-log", action="store_true", help="enable uvicorn per-request access log")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model = os.path.expanduser(args.model)
    served = args.served_model_name or os.path.basename(os.path.normpath(model))
    engine = AsyncLLMEngine.from_model(
        model,
        max_waiting_requests=args.max_waiting_requests,
        metrics=ServerMetrics(served),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        scheduler_policy=args.scheduler_policy,
    )
    app = create_app(engine, served_model_name=served, model_path=model)
    # On SIGINT/SIGTERM: stop accepting, give open streams 10 s, then the app lifespan
    # aborts whatever is still running and releases the GPU.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=args.access_log,
                timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
