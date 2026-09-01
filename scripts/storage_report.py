#!/usr/bin/env python3
"""Storage, weight-traffic, and stored-mean accounting for T-VELCRO.

Answers, per model and bypass fraction (Freddy notes 2/3/4):
  - weights removed by structural slicing (params and bytes)
  - stored-mean constant tensors that replace them (params and bytes)
  - NET storage saving = removed weights - stored means  (sliced deployment)
  - per-profile mean bytes for the hook deployment (one binary + K profiles)
  - analytical weight-memory-traffic saving per forward at batch 1
    (each surviving weight is read once per forward)

Everything here is exact parameter arithmetic from the architecture shapes;
cross-check against measured `params` in bench_latency JSONs on the pod.

Usage:  python3 scripts/storage_report.py [--json results/storage_report.json]
        # optional exact seq-len override from a real mean file:
        python3 scripts/storage_report.py --mean_npy path/to/block0_head3_mean.npy
"""

import argparse
import json

# blocks, heads/block, C hidden, d_h, list of (seq_len, n_blocks) groups, total model params
MODELS = {
    # 224px/16 -> 196 patches + cls
    "vit_base": dict(C=768, d_h=64, groups=[(197, 12)], H=12, total_params=86.6e6, bench_dtype="fp32"),
    # 518px/14 -> 37^2 = 1369 patches + cls
    "dinov2_giant": dict(C=1536, d_h=64, groups=[(1370, 40)], H=24, total_params=1.14e9, bench_dtype="fp32"),
    # 224px/16 -> 196 patches + cls + 4 registers (VERIFY on pod: mean .npy shape)
    "dinov3_7b": dict(C=4096, d_h=128, groups=[(201, 40)], H=32, total_params=6.7e9, bench_dtype="fp16"),
    # SAM ViT-H encoder: 28 windowed (14x14=196) + 4 global (64x64=4096) blocks
    "sam_vit_h": dict(C=1280, d_h=80, groups=[(196, 28), (4096, 4)], H=16, total_params=637e6, bench_dtype="fp32"),
}

FRACS = [0.125, 0.25, 0.375, 0.5]
DTYPE_BYTES = {"fp32": 4, "fp16": 2}


def fmt(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n:.0f} B"
        n /= 1024


def report(name, m, frac, mean_dtype="fp16", weight_dtype="fp16"):
    C, d_h, H = m["C"], m["d_h"], m["H"]
    t_b = int(frac * H)  # heads bypassed per block (floor, as in the paper)
    removed_w = 0  # weight params removed
    mean_vals = 0  # stored mean values
    for S_b, n_blocks in m["groups"]:
        per_head_w = 4 * C * d_h + 3 * d_h  # QKV rows (+bias) and W_O columns
        removed_w += n_blocks * t_b * per_head_w
        mean_vals += n_blocks * t_b * S_b * d_h
    wb, mb = DTYPE_BYTES[weight_dtype], DTYPE_BYTES[mean_dtype]
    removed_bytes = removed_w * wb
    mean_bytes = mean_vals * mb
    net_bytes = removed_bytes - mean_bytes
    total_bytes = m["total_params"] * wb
    return {
        "model": name, "frac": frac, "heads_bypassed_per_block": t_b,
        "weight_dtype": weight_dtype, "mean_dtype": mean_dtype,
        "removed_weight_params": removed_w,
        "stored_mean_values": mean_vals,
        "removed_weight_bytes": removed_bytes,
        "stored_mean_bytes": mean_bytes,          # note 4: the constant tensors
        "net_storage_saving_bytes": net_bytes,    # note 3: net, sliced deployment
        "net_saving_pct_of_model": 100.0 * net_bytes / total_bytes,
        "mean_eats_pct_of_removed": 100.0 * mean_bytes / removed_bytes,
        # note 2 (analytical): batch-1 weight traffic == resident weight bytes,
        # and the mean constants must be read back in once per forward
        "weight_traffic_saving_bytes_per_fwd": net_bytes,
        "hook_profile_bytes": mean_bytes,         # per-domain profile (frozen means only)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None)
    p.add_argument("--mean_npy", default=None,
                   help="optional real mean file; prints its shape to verify seq len")
    args = p.parse_args()

    if args.mean_npy:
        import numpy as np
        a = np.load(args.mean_npy, mmap_mode="r")
        print(f"mean file shape={a.shape} dtype={a.dtype}  (S x d_h to verify against MODELS)")

    rows = []
    for name, m in MODELS.items():
        for frac in FRACS:
            rows.append(report(name, m, frac))

    hdr = (f"{'model':<14}{'frac':>6}{'t_b':>4} | {'removed W':>10}{'means':>10}"
           f"{'NET saved':>11}{'% model':>8}{'mean/rm%':>9} | {'profile':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<14}{r['frac']:>6}{r['heads_bypassed_per_block']:>4} | "
              f"{fmt(r['removed_weight_bytes']):>10}{fmt(r['stored_mean_bytes']):>10}"
              f"{fmt(r['net_storage_saving_bytes']):>11}{r['net_saving_pct_of_model']:>7.1f}%"
              f"{r['mean_eats_pct_of_removed']:>8.1f}% | {fmt(r['hook_profile_bytes']):>9}")
    if args.json:
        from pathlib import Path
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
