#!/usr/bin/env python3
"""
report_profile_size.py -- Storage and overhead report for SAM VELCRO profiles.

Reports:
  - total profile size on disk (all .npy files under the profile dir)
  - frozen-heads-only size (mean files referenced by a freeze manifest)
  - FP16-converted size for both (what the profile would cost stored as float16)
  - micro-benchmark of freeze hook attach/detach time (CPU-only, no GPU needed)

Usage (from repo root):
  python3 sam/report_profile_size.py --profile_dir results/sam_main_sweep/profile
  python3 sam/report_profile_size.py --profile_dir results/sam_main_sweep/profile \
      --freeze_manifest results/sam_main_sweep/freeze_kvasir_f0.25/freeze_manifest.json

Outputs a human-readable summary to stdout and JSON to --out
(default: <profile_dir>/profile_size_report.json).
"""

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np

NPY_HEADER_BYTES = 128  # ponytail: standard .npy v1 header size, close enough for a report


def npy_sizes(paths):
    """Return (disk_bytes, fp16_bytes) for a list of .npy files."""
    disk = 0
    fp16 = 0
    for p in paths:
        disk += p.stat().st_size
        arr = np.load(p, mmap_mode="r")
        fp16 += arr.size * 2 + NPY_HEADER_BYTES
    return disk, fp16


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def benchmark_hooks(manifest, reps=20):
    """Time mean loading, hook manager init, and attach/detach on CPU.

    Uses a dummy module tree shaped like SAM's image_encoder so no
    checkpoint or GPU is needed; hook attach/detach cost does not depend
    on the module's weights.
    """
    import torch
    import torch.nn as nn

    from sam_velcro_pkg.constants import NUM_BLOCKS
    from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest
    from sam_velcro_pkg.hooks import FreezeHookManager

    t0 = time.perf_counter()
    frozen = load_frozen_heads_from_manifest(manifest)
    load_s = time.perf_counter() - t0

    model = types.SimpleNamespace(image_encoder=types.SimpleNamespace(blocks=[
        types.SimpleNamespace(attn=types.SimpleNamespace(proj=nn.Identity()))
        for _ in range(NUM_BLOCKS)
    ]))

    t0 = time.perf_counter()
    mgr = FreezeHookManager(model, frozen, torch.device("cpu"))
    init_s = time.perf_counter() - t0

    attach_ms = []
    detach_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mgr.register()
        attach_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        mgr.remove()
        detach_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "num_frozen_heads": len(frozen),
        "mean_load_seconds": round(load_s, 4),
        "manager_init_seconds": round(init_s, 4),
        "attach_ms_mean": round(float(np.mean(attach_ms)), 4),
        "attach_ms_std": round(float(np.std(attach_ms)), 4),
        "detach_ms_mean": round(float(np.mean(detach_ms)), 4),
        "detach_ms_std": round(float(np.std(detach_ms)), 4),
        "reps": reps,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile size and hook overhead report")
    parser.add_argument("--profile_dir", required=True,
                        help="Profile directory (contains <group>/ subdirs with .npy files)")
    parser.add_argument("--freeze_manifest", default=None,
                        help="Optional freeze_manifest.json for frozen-only size + hook benchmark")
    parser.add_argument("--reps", type=int, default=20,
                        help="Repetitions for the attach/detach benchmark")
    parser.add_argument("--out", default=None,
                        help="JSON output path (default: <profile_dir>/profile_size_report.json)")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        raise SystemExit(f"ERROR: profile dir not found: {profile_dir}")

    npy_files = sorted(profile_dir.rglob("*.npy"))
    if not npy_files:
        raise SystemExit(f"ERROR: no .npy files under {profile_dir}")

    total_disk, total_fp16 = npy_sizes(npy_files)

    report = {
        "profile_dir": str(profile_dir),
        "num_npy_files": len(npy_files),
        "total_disk_bytes": total_disk,
        "total_fp16_bytes": total_fp16,
    }

    print("=" * 64)
    print("SAM VELCRO Profile Size Report")
    print("=" * 64)
    print(f"Profile dir:        {profile_dir}")
    print(f"  .npy files:       {len(npy_files)}")
    print(f"  Total on disk:    {fmt_bytes(total_disk)}")
    print(f"  FP16-converted:   {fmt_bytes(total_fp16)} "
          f"({total_fp16 / total_disk * 100:.0f}% of current)")

    if args.freeze_manifest:
        from sam_velcro_pkg.freeze import load_manifest, resolve_profile_path

        manifest = load_manifest(Path(args.freeze_manifest))
        mean_paths = []
        missing = 0
        for entry in manifest["frozen_heads"]:
            p = resolve_profile_path(manifest, entry["mean_path"])
            if p.exists():
                mean_paths.append(p)
            else:
                missing += 1
        frozen_disk, frozen_fp16 = npy_sizes(mean_paths)

        report["frozen"] = {
            "manifest": str(args.freeze_manifest),
            "num_frozen_heads": len(manifest["frozen_heads"]),
            "num_mean_files_found": len(mean_paths),
            "num_mean_files_missing": missing,
            "mean_files_disk_bytes": frozen_disk,
            "mean_files_fp16_bytes": frozen_fp16,
        }

        print()
        print(f"Freeze manifest:    {args.freeze_manifest}")
        missing_note = f" ({missing} mean files missing)" if missing else ""
        print(f"  Frozen heads:     {len(manifest['frozen_heads'])}{missing_note}")
        print(f"  Frozen-only size: {fmt_bytes(frozen_disk)} (mean files only, what eval loads)")
        print(f"  FP16-converted:   {fmt_bytes(frozen_fp16)}")

        bench = benchmark_hooks(manifest, reps=args.reps)
        report["hook_benchmark"] = bench

        print()
        print(f"Hook micro-benchmark (CPU, {bench['reps']} reps):")
        print(f"  Mean .npy load:   {bench['mean_load_seconds']:.4f} s "
              f"({bench['num_frozen_heads']} heads)")
        print(f"  Manager init:     {bench['manager_init_seconds']:.4f} s (numpy -> torch)")
        print(f"  Hook attach:      {bench['attach_ms_mean']:.3f} +/- {bench['attach_ms_std']:.3f} ms")
        print(f"  Hook detach:      {bench['detach_ms_mean']:.3f} +/- {bench['detach_ms_std']:.3f} ms")

    out_path = Path(args.out) if args.out else profile_dir / "profile_size_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(f"JSON report: {out_path}")


if __name__ == "__main__":
    main()
