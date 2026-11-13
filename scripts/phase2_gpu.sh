#!/usr/bin/env bash
# Phase 2 GPU checks, one command. Run on the pod from the repo root after scripts/pod_setup.sh:
#     bash scripts/phase2_gpu.sh 2>&1 | tee benchmarks/results/raw/02_phase2_gpu.log
# Steps (one GPU process at a time):
#   1. full pytest (CPU tests + real-tokenizer tests)
#   2. offline regression: upstream bench.py x3 with the Phase 2 engine changes
#   3. greedy determinism: alone vs batched
#   4. server transcript: curl + official openai client + /metrics excerpt
#   5. vllm bench serve smoke run (low rate; proof it runs end to end, not a result)
#   6. serving overhead: offline generate() vs online at saturation (+ py-spy if available)
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/env.sh
PY=/root/.venv/bin/python
VLLM=/root/.venv-vllm/bin/vllm
MODEL=/workspace/hf-models/Qwen3-0.6B
NAME=Qwen3-0.6B
PORT=8000
OUT=benchmarks/results
RAW=$OUT/raw
mkdir -p "$RAW"
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S))"; }
gpu_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

step "0. environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git log --oneline -1
gpu_idle || { echo "GPU busy, aborting"; exit 1; }

step "1. pytest"
$PY -m pytest -q 2>&1 | tee "$RAW/02_pytest.txt" | tail -3

step "2. offline regression (upstream bench.py x3)"
$PY benchmarks/run_upstream_offline.py --runs 3 --out "$OUT/02_offline_regression.json"

step "3. greedy determinism"
$PY benchmarks/greedy_determinism.py --model "$MODEL" --out "$OUT/02_greedy_determinism.json"

step "4. server transcript"
gpu_idle
$PY -m nanovllm.serve --model "$MODEL" --served-model-name "$NAME" --port $PORT \
    --scheduler-policy prefill_first > "$RAW/02_server.log" 2>&1 &
SERVER=$!
trap 'kill -INT $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf localhost:$PORT/health >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -40 "$RAW/02_server.log"; exit 1; }
  sleep 2
done
{
  echo '$ curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/health'
  curl -s -o /dev/null -w "%{http_code}\n" localhost:$PORT/health
  echo; echo '$ curl -s localhost:8000/v1/models'
  curl -s localhost:$PORT/v1/models; echo
  echo; echo '$ curl -s localhost:8000/v1/completions -d {"prompt":"The capital of France is","max_tokens":12,"temperature":0}'
  curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
       -d '{"model":"'$NAME'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'; echo
  echo; echo '$ curl -sN localhost:8000/v1/completions -d {"prompt":"用中文说你好，然后加一个emoji：","max_tokens":10,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
  curl -sN localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
       -d '{"model":"'$NAME'","prompt":"用中文说你好，然后加一个emoji：","max_tokens":10,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
  echo; echo '$ curl -s localhost:8000/v1/chat/completions -d {"messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":40,"temperature":0}'
  curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"'$NAME'","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":40,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'; echo
  echo; echo '$ curl -s -w "%{http_code}\n" localhost:8000/v1/completions -d {"prompt":"x","n":2}'
  curl -s -w "%{http_code}\n" localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"x","n":2}'
  echo; echo '$ python benchmarks/openai_client_demo.py'
  $PY benchmarks/openai_client_demo.py --base-url http://127.0.0.1:$PORT/v1 --model $NAME
} 2>&1 | tee "$OUT/02_transcript.txt"

step "5. vllm bench serve smoke (low rate)"
$VLLM bench serve --backend openai --base-url http://127.0.0.1:$PORT --endpoint /v1/completions \
    --model $NAME --tokenizer "$MODEL" --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 40 --request-rate 2 --ignore-eos --seed 0 --disable-tqdm \
    --save-result --result-dir "$RAW" --result-filename 02_bench_smoke.json 2>&1 | tee "$RAW/02_bench_smoke.log" | tail -30
{
  echo; echo '$ curl -s localhost:8000/metrics | grep -E "^nanovllm_(engine_steps|kv_cache|preemptions|prefix_cache|requests_finished|generation_tokens|time_to_first_token_seconds_(count|sum)|inter_token_latency_seconds_(count|sum))"'
  curl -s localhost:$PORT/metrics | grep -E "^nanovllm_(engine_steps|kv_cache|preemptions|prefix_cache|requests_finished|generation_tokens|time_to_first_token_seconds_(count|sum)|inter_token_latency_seconds_(count|sum))"
} 2>&1 | tee -a "$OUT/02_transcript.txt"

kill -INT $SERVER; wait $SERVER || true; trap - EXIT
for i in $(seq 1 30); do gpu_idle && break; sleep 1; done
gpu_idle || { echo "GPU still busy after server stop"; exit 1; }
echo "server stopped cleanly (graceful shutdown), GPU idle"

step "6. serving overhead"
PROFILE=""
if py-spy --version >/dev/null 2>&1 && py-spy dump --pid $$ >/dev/null 2>&1; then PROFILE="--profile"; fi
echo "py-spy profiling: ${PROFILE:-unavailable}"
$PY benchmarks/serving_overhead.py --model "$MODEL" --served-model-name $NAME --port $PORT --runs 2 $PROFILE

step "done"
ls -la "$OUT" "$RAW"
