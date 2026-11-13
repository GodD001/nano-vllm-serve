"""Phase 3 gate: does flash_attn_varlen_func with a block_table handle seqlen_q = 1 rows
(a decode sequence expressed as a 1-token chunk over its cached prefix), alone and mixed
with a prefill chunk in the same call?

Compares, on random bf16 data with Qwen3-0.6B attention shapes and nano-vLLM's block size:
  ref_decode : flash_attn_with_kvcache (upstream decode path)
  varlen     : flash_attn_varlen_func(..., block_table) with cu_seqlens_q steps of 1
  torch      : naive fp32 attention over the gathered paged KV (ground truth)
and a mixed call (decodes + one prefill chunk) against separate calls.

    python benchmarks/verify_mixed_varlen.py --out benchmarks/results/03_verify_mixed_varlen.json
"""
import argparse
import json
import os

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

H_Q, H_KV, D, BLOCK = 16, 8, 128, 256


def gather(cache, table, n):
    """[num_blocks, BLOCK, H, D] + block ids -> first n positions, [n, H, D]"""
    return cache[table].reshape(-1, cache.size(2), cache.size(3))[:n]


def torch_attn(q, k, v, q_start):
    """q: [Lq, Hq, D] at absolute positions q_start..; k, v: [Lk, Hkv, D]. Causal, GQA. fp32."""
    rep = H_Q // H_KV
    k = k.float().repeat_interleave(rep, dim=1)
    v = v.float().repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), k) / D ** 0.5
    qpos = torch.arange(q_start, q_start + q.size(0), device=q.device)[:, None]
    kpos = torch.arange(k.size(0), device=q.device)[None, :]
    s = s.masked_fill((kpos > qpos)[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results/03_verify_mixed_varlen.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev, dt = "cuda", torch.bfloat16
    num_blocks = 64
    k_cache = torch.randn(num_blocks, BLOCK, H_KV, D, device=dev, dtype=dt)
    v_cache = torch.randn(num_blocks, BLOCK, H_KV, D, device=dev, dtype=dt)
    perm = torch.randperm(num_blocks).tolist()

    # decode sequences: context lengths cover block boundaries (1, 255, 256, 257, ...)
    ctx = [1, 37, 255, 256, 257, 511, 600, 1500]
    max_blocks = 8
    tables, used = [], 0
    for L in ctx:
        nb = -(-L // BLOCK)
        tables.append(perm[used:used + nb] + [0] * (max_blocks - nb))
        used += nb
    # one prefill chunk: 300 cached tokens + 200 new query tokens (prefix-cache style)
    pf_cached, pf_new = 300, 200
    pf_nb = -(-(pf_cached + pf_new) // BLOCK)
    pf_table = perm[used:used + pf_nb] + [0] * (max_blocks - pf_nb)

    B = len(ctx)
    q_dec = torch.randn(B, H_Q, D, device=dev, dtype=dt)
    q_pf = torch.randn(pf_new, H_Q, D, device=dev, dtype=dt)
    bt_dec = torch.tensor(tables, dtype=torch.int32, device=dev)
    lens = torch.tensor(ctx, dtype=torch.int32, device=dev)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)  # noqa: E731

    with torch.inference_mode():
        ref = flash_attn_with_kvcache(q_dec.unsqueeze(1), k_cache, v_cache, cache_seqlens=lens,
                                      block_table=bt_dec, causal=True).squeeze(1)
        cu_q = i32(list(range(B + 1)))
        cu_k = i32([0] + torch.tensor(ctx).cumsum(0).tolist())
        varlen = flash_attn_varlen_func(q_dec, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                        max_seqlen_q=1, max_seqlen_k=max(ctx), causal=True, block_table=bt_dec)
        truth = torch.stack([torch_attn(q_dec[i:i + 1], gather(k_cache, tables[i], L), gather(v_cache, tables[i], L), L - 1)[0]
                             for i, L in enumerate(ctx)])

        # mixed: the decodes + the prefill chunk in one call
        q_mix = torch.cat([q_dec, q_pf])
        cu_q_mix = i32(list(range(B + 1)) + [B + pf_new])
        cu_k_mix = i32(cu_k.tolist() + [cu_k[-1].item() + pf_cached + pf_new])
        bt_mix = torch.cat([bt_dec, i32([pf_table])])
        mixed = flash_attn_varlen_func(q_mix, k_cache, v_cache, cu_seqlens_q=cu_q_mix, cu_seqlens_k=cu_k_mix,
                                       max_seqlen_q=pf_new, max_seqlen_k=max(max(ctx), pf_cached + pf_new),
                                       causal=True, block_table=bt_mix)
        pf_alone = flash_attn_varlen_func(q_pf, k_cache, v_cache, cu_seqlens_q=i32([0, pf_new]),
                                          cu_seqlens_k=i32([0, pf_cached + pf_new]), max_seqlen_q=pf_new,
                                          max_seqlen_k=pf_cached + pf_new, causal=True, block_table=i32([pf_table]))
        pf_truth = torch_attn(q_pf, gather(k_cache, pf_table, pf_cached + pf_new),
                              gather(v_cache, pf_table, pf_cached + pf_new), pf_cached)

    def maxdiff(a, b):
        return (a.float() - b.float()).abs().max().item()

    result = {
        "shapes": {"num_q_heads": H_Q, "num_kv_heads": H_KV, "head_dim": D, "block_size": BLOCK, "dtype": "bfloat16"},
        "decode_context_lens": ctx,
        "prefill_chunk": {"cached": pf_cached, "new": pf_new},
        "max_abs_diff": {
            "varlen_q1_vs_flash_decode": maxdiff(varlen, ref),
            "varlen_q1_vs_torch_fp32": maxdiff(varlen, truth),
            "flash_decode_vs_torch_fp32": maxdiff(ref, truth),
            "mixed_decode_rows_vs_flash_decode": maxdiff(mixed[:B], ref),
            "mixed_prefill_rows_vs_prefill_alone": maxdiff(mixed[B:], pf_alone),
            "mixed_prefill_rows_vs_torch_fp32": maxdiff(mixed[B:], pf_truth),
        },
        "torch": torch.__version__,
        "flash_attn": __import__("flash_attn").__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    # bf16 outputs of the same math differ by rounding; fp32 ground truth bounds both kernels.
    d = result["max_abs_diff"]
    tol = 2 * max(d["flash_decode_vs_torch_fp32"], 1e-2)
    result["tolerance"] = {"rule": "varlen/mixed error vs fp32 must be <= 2x the upstream decode kernel's (min 1e-2)",
                           "value": tol}
    result["pass"] = all(d[k] <= tol for k in ("varlen_q1_vs_torch_fp32", "mixed_prefill_rows_vs_torch_fp32")) \
        and d["mixed_decode_rows_vs_flash_decode"] <= tol
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
