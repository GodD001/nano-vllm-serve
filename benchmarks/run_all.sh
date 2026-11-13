#!/usr/bin/env bash
# Phase 4: run the whole approved benchmark matrix with one command (on the pod, repo root):
#     bash benchmarks/run_all.sh 2>&1 | tee benchmarks/results/raw/phase4/run_all.log
# Resumable: points whose raw JSON already exists are skipped, so re-running after a pod
# interruption continues where it stopped. Requires benchmarks/matrix.json with rates filled
# in (Checkpoint 4a).
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/env.sh
PY=/root/.venv/bin/python
MODEL=$($PY -c "import json; print(json.load(open('benchmarks/matrix.json'))['model'])")
mkdir -p benchmarks/results/raw/phase4
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy"; exit 1; }
grep -q '"_status": "DRAFT' benchmarks/matrix.json && { echo "matrix.json is still a DRAFT (Checkpoint 4a)"; exit 1; }

echo "== workloads ($(date -u +%H:%M:%S))"
[ -f benchmarks/workloads/w2_mixed.jsonl ] || $PY benchmarks/make_workloads.py --model "$MODEL"
$PY - <<'EOF'
import hashlib, json
meta = json.load(open("benchmarks/workloads/w2_mixed.meta.json"))
h = hashlib.sha256(open("benchmarks/workloads/w2_mixed.jsonl", "rb").read()).hexdigest()
assert h == meta["sha256"], f"w2_mixed.jsonl differs from its committed meta ({h} != {meta['sha256']})"
print("w2_mixed.jsonl sha256 ok")
EOF

echo "== matrix ($(date -u +%H:%M:%S))";    $PY benchmarks/run_matrix.py --run
echo "== knob sweep ($(date -u +%H:%M:%S))"; $PY benchmarks/run_matrix.py --knob
echo "== report ($(date -u +%H:%M:%S))";     $PY benchmarks/report.py
echo "== done ($(date -u +%H:%M:%S))"
