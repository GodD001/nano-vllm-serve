#!/usr/bin/env bash
# Phase 3 GPU checks, one command (run on the pod from the repo root after scripts/pod_setup.sh):
#     bash scripts/phase3_gpu.sh 2>&1 | tee benchmarks/results/raw/03_phase3_gpu.log
#   1. pytest (CPU + real-tokenizer tests)
#   2. kernel gate: flash_attn_varlen_func + block_table with seqlen_q = 1, alone and mixed
#   3. logit equivalence: one mixed step vs upstream's separate decode + prefill steps
#   4. greedy agreement: prefill_first vs decode_first, staggered arrivals
#   5. step time: decode-only (graph / eager) vs mixed vs separate, by batch size
#   6. offline regression: bench.py workload under both policies vs the Phase 0 baseline
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/env.sh
PY=/root/.venv/bin/python
MODEL=/workspace/hf-models/Qwen3-0.6B
OUT=benchmarks/results
mkdir -p "$OUT/raw"
step() { echo; echo "=================== $* ($(date -u +%H:%M:%S))"; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy"; exit 1; }
git log --oneline -1

step "1. pytest";             $PY -m pytest -q -p no:cacheprovider 2>&1 | tee "$OUT/raw/03_pytest.txt" | tail -2
step "2. varlen kernel gate"; $PY benchmarks/verify_mixed_varlen.py --out "$OUT/03_verify_mixed_varlen.json" | tail -14
step "3. logit equivalence";  $PY benchmarks/logit_equivalence.py --model "$MODEL" --out "$OUT/03_logit_equivalence.json"
step "4. greedy agreement";   $PY benchmarks/policy_agreement.py --model "$MODEL" --out "$OUT/03_policy_agreement.json" \
                                  | grep -E '"(identical_sequences|token_agreement_before_first_divergence|num_requests)"'
step "5. step time";          $PY benchmarks/step_time.py --model "$MODEL" --out "$OUT/03_step_time.json"
step "6. offline regression"; $PY benchmarks/bench_policy.py --runs 3 --out "$OUT/03_offline_regression.json"
step "done"
