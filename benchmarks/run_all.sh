#!/usr/bin/env bash
# Phase 4: the whole approved benchmark plan with one command (on the pod, repo root):
#     bash benchmarks/run_all.sh 2>&1 | tee benchmarks/results/raw/phase4/run_all.log
# Resumable: points whose raw JSON already exists are skipped, so re-running after a pod
# interruption continues where it stopped. Steps:
#   0.6B  W3 saturation -> W3 rates (fixed rule) -> matrix W1/W2/W3 -> knob sweep -> report
#   1.7B  download -> W2 saturation -> rates (same rule) -> matrix W2 -> report
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/env.sh
PY=/root/.venv/bin/python
M06=benchmarks/matrix.json
M17=benchmarks/matrix_qwen3_1p7b.json
mkdir -p benchmarks/results/raw/phase4
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S))"; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy"; exit 1; }
for m in $M06 $M17; do grep -q '"_status": "DRAFT' $m && { echo "$m is still a DRAFT (Checkpoint 4a)"; exit 1; }; done
git log --oneline -1

step "workloads"
$PY benchmarks/make_workloads.py --model /workspace/hf-models/Qwen3-0.6B | grep -E "matches|differs|sha256"

step "0.6B: W3 saturation + rates"
$PY benchmarks/run_matrix.py --config $M06 --find-saturation --workloads W3_shared_prefix
$PY benchmarks/run_matrix.py --config $M06 --set-rates --workloads W3_shared_prefix
step "0.6B: matrix W1 / W2 / W3"
$PY benchmarks/run_matrix.py --config $M06 --run --workloads W1_short,W2_mixed,W3_shared_prefix
step "0.6B: knob sweep"
$PY benchmarks/run_matrix.py --config $M06 --knob
step "0.6B: report"
$PY benchmarks/report.py

step "1.7B: model"
[ -f /workspace/hf-models/Qwen3-1.7B/config.json ] || /root/.venv/bin/hf download Qwen/Qwen3-1.7B \
    --local-dir /workspace/hf-models/Qwen3-1.7B > /dev/null
$PY -c "import json; c=json.load(open('/workspace/hf-models/Qwen3-1.7B/config.json')); \
print({k: c[k] for k in ('num_hidden_layers','num_key_value_heads','head_dim','torch_dtype') if k in c})"
step "1.7B: W2 saturation + rates"
$PY benchmarks/run_matrix.py --config $M17 --find-saturation
$PY benchmarks/run_matrix.py --config $M17 --set-rates
step "1.7B: matrix W2"
$PY benchmarks/run_matrix.py --config $M17 --run
step "1.7B: report"
$PY benchmarks/report.py --raw benchmarks/results/raw/phase4_qwen3_1p7b --out benchmarks/results/summary_qwen3_1p7b.md \
    --img docs/images/qwen3_1p7b --title "Phase 4 headline comparison on Qwen3-1.7B (W2)"
step "done"
