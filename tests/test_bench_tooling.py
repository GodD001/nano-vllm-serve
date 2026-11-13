"""Phase 4 tooling (no GPU): server / bench command construction and the report pipeline."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
import report  # noqa: E402
import run_matrix  # noqa: E402

CFG = json.load(open(os.path.join(os.path.dirname(run_matrix.__file__), "matrix.json")))


def test_equal_kv_capacity_for_all_systems():
    nano = run_matrix.server_cmd(CFG, CFG["systems"]["A_prefill_first"])
    vllm = run_matrix.server_cmd(CFG, CFG["systems"]["C_vllm"])
    nano_blocks = int(nano[nano.index("--max-kvcache-blocks") + 1])
    vllm_blocks = int(vllm[vllm.index("--num-gpu-blocks-override") + 1])
    assert nano_blocks * 256 == vllm_blocks * 16 == CFG["kv_cache_tokens"]
    for cmd in (nano, vllm):
        assert cmd[cmd.index("--max-model-len") + 1] == str(CFG["max_model_len"])
        assert cmd[cmd.index("--served-model-name") + 1] == CFG["served_model_name"]
    b = run_matrix.server_cmd(CFG, CFG["systems"]["B_decode_first"], ["--max-prefill-tokens-per-step", "512"])
    assert "decode_first" in b and b[-2:] == ["--max-prefill-tokens-per-step", "512"]


def test_bench_args_identical_across_systems():
    w = CFG["workloads"]["W2_mixed"]
    args = run_matrix.bench_args(CFG, w, 3.5, 500)
    for flag, value in [("--num-prompts", "500"), ("--request-rate", "3.5"), ("--seed", "0")]:
        assert args[args.index(flag) + 1] == value
    assert "--ignore-eos" in args and "--skip-chat-template" in args


def _write_point(raw, system, workload, rate, ttft, itl, out_tokens, server_ttft, server_itl):
    d = raw / system / workload
    d.mkdir(parents=True, exist_ok=True)
    bench = {"request_throughput": 2.0, "output_throughput": 500.0, "completed": 500, "failed": 0,
             "total_output_tokens": out_tokens, "mean_ttft_ms": ttft, "mean_itl_ms": itl,
             "p50_ttft_ms": ttft, "p90_ttft_ms": ttft * 1.5, "p99_ttft_ms": ttft * 2,
             "p50_itl_ms": itl, "p90_itl_ms": itl * 1.2, "p99_itl_ms": itl * 2,
             "p50_tpot_ms": itl, "p99_tpot_ms": itl * 1.5, "p99_e2el_ms": 9000.0}
    json.dump(bench, open(d / f"rate_{rate}.json", "w"))
    if system.startswith("C"):
        counters = {'vllm:num_preemptions_total{engine="0"}': 3.0,
                    'vllm:time_to_first_token_seconds_sum{engine="0"}': server_ttft * 500,
                    'vllm:time_to_first_token_seconds_count{engine="0"}': 500.0,
                    'vllm:inter_token_latency_seconds_sum{engine="0"}': server_itl * 1000,
                    'vllm:inter_token_latency_seconds_count{engine="0"}': 1000.0}
        gauges = {'vllm:kv_cache_usage_perc{engine="0"}': {"max": 0.8, "mean": 0.5, "n": 10}}
    else:
        counters = {"nanovllm_preemptions_total": 1.0,
                    "nanovllm_time_to_first_token_seconds_sum": server_ttft * 500,
                    "nanovllm_time_to_first_token_seconds_count": 500.0,
                    "nanovllm_inter_token_latency_seconds_sum": server_itl * 1000,
                    "nanovllm_inter_token_latency_seconds_count": 1000.0,
                    'nanovllm_step_batch_size_sum{type="decode"}': 640.0,
                    'nanovllm_step_batch_size_count{type="decode"}': 10.0}
        gauges = {"nanovllm_kv_cache_blocks_used": {"max": 120.0, "mean": 60.0, "n": 10},
                  "nanovllm_kv_cache_blocks_total": {"max": 160.0, "mean": 160.0, "n": 10}}
    json.dump({"counters_delta": counters, "gauges": gauges}, open(d / f"rate_{rate}.server_metrics.json", "w"))


def test_report_end_to_end(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for system in ("A_prefill_first", "B_decode_first", "C_vllm"):
        for rate in (1.0, 2.0):
            _write_point(raw, system, "W1_short", rate, 100.0, 20.0, 128000, 0.1, 0.02)
    _write_point(raw, "C_vllm", "W2_mixed", 1.0, 300.0, 25.0, 128000, 0.3, 0.025)
    _write_point(raw, "A_prefill_first", "W2_mixed", 1.0, 300.0, 25.0, 127999, 0.3, 0.025)   # mismatch
    monkeypatch.setattr(report, "RAW", str(raw))
    monkeypatch.setattr(report, "OUT_MD", str(tmp_path / "summary.md"))
    monkeypatch.setattr(report, "IMG", str(tmp_path / "images"))
    report.main([])
    md = (tmp_path / "summary.md").read_text()
    assert "## W1_short" in md and "## W2_mixed" in md
    assert md.count("| A: nano prefill_first (upstream policy) |") == 3
    assert "| 1.00 | 1.00 |" in md                              # client/server ratios in the sanity table
    assert "75%" in md and "80%" in md                           # KV max for nano (120/160) and vLLM
    assert "**Output-token totals differ:**" in md and "W2_mixed" in md
    pytest.importorskip("matplotlib")
    assert len(list((tmp_path / "images").glob("*.png"))) == 6   # 2 workloads x 3 plots
