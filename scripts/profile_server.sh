#!/usr/bin/env bash
# Profile the server under the Phase 2d saturation workload with py-spy.
# The pod has no SYS_PTRACE (ptrace_scope=1), so py-spy cannot attach to a running process;
# it launches the server as its own child instead.
#
#   bash scripts/profile_server.sh gil  TAG   # sample only the thread holding the GIL
#   bash scripts/profile_server.sh cpu  TAG   # sample all non-idle threads
# Output: benchmarks/results/raw/02_pyspy_<mode>_<TAG>.txt (collapsed stacks, one per line)
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/env.sh
MODE=${1:-gil}
TAG=${2:-before}
PY=/root/.venv/bin/python
VLLM=/root/.venv-vllm/bin/vllm
MODEL=/workspace/hf-models/Qwen3-0.6B
NAME=Qwen3-0.6B
PORT=8000
RAW=benchmarks/results/raw
OUTF=$RAW/02_pyspy_${MODE}_${TAG}.txt
mkdir -p "$RAW"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy"; exit 1; }

FLAGS="--threads --rate 200 --format raw --nonblocking"
[ "$MODE" = gil ] && FLAGS="$FLAGS --gil"
/root/.venv/bin/py-spy record $FLAGS -o "$OUTF" -- \
    $PY -m nanovllm.serve --model "$MODEL" --served-model-name $NAME --port $PORT \
    --scheduler-policy prefill_first > "$RAW/02_pyspy_${MODE}_${TAG}_server.log" 2>&1 &
SPY=$!
for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 2; done
$VLLM bench serve --backend openai --base-url http://127.0.0.1:$PORT --endpoint /v1/completions \
    --model $NAME --tokenizer "$MODEL" --dataset-name random --random-input-len 512 --random-output-len 256 \
    --random-range-ratio 0.0 --num-prompts 256 --request-rate inf --ignore-eos --seed 0 --num-warmups 16 \
    --disable-tqdm --save-result --result-dir "$RAW" --result-filename "02_pyspy_${MODE}_${TAG}_bench.json" \
    > "$RAW/02_pyspy_${MODE}_${TAG}_bench.log" 2>&1
grep -E "Output token throughput|Median ITL" "$RAW/02_pyspy_${MODE}_${TAG}_bench.log"
pkill -INT -f "nanovllm.serve --model" || true
wait $SPY || true
for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
echo "wrote $OUTF ($(wc -l < "$OUTF") stacks)"
