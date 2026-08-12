#!/usr/bin/env python3
"""
plot_sweep_results.py

Reads SAM VELCRO sweep results and generates:
  - Per-group IoU vs FLOP reduction PNGs
  - Combined all-groups IoU PNG
  - Cross-group variance comparison histograms
  - Before/after variance plots
  - Windowed vs global rank comparison
  - Domain specialization matrix
  - Summary report (text table)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sam_velcro_pkg.constants import DOMAIN_GROUPS, GLOBAL_ATTN_INDEXES, NUM_BLOCKS, NUM_HEADS
from sam_velcro_pkg.flops import compute_flop_reduction_from_manifest
from sam_velcro_pkg.plotting import (
    plot_before_after_variance,
    plot_cross_group_variance_comparison,
    plot_domain_specialization_matrix,
    plot_iou_vs_flop_reduction,
    plot_windowed_vs_global_rank,
)


def load_sweep_data(results_dir: Path, groups: list, fractions: list) -> list:
    sweep_results = []

    for group in groups:
        for frac in fractions:
            frac_str = f"{frac}"
            eval_dir = results_dir / f"eval_{group}_f{frac_str}"

            for eval_group_dir in eval_dir.iterdir() if eval_dir.exists() else []:
                if not eval_group_dir.is_dir():
                    continue
                result_path = eval_group_dir / "eval_results.json"
                if not result_path.exists():
                    continue

                with open(result_path) as f:
                    result = json.load(f)

                manifest_path = results_dir / f"freeze_{group}_f{frac_str}" / "freeze_manifest.json"
                n_frozen = 0
                flop_red = 0.0
                flop_red_encoder = 0.0
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    n_frozen = manifest.get("summary", {}).get("frozen", 0)
                    flop_info = compute_flop_reduction_from_manifest(manifest)
                    flop_red = flop_info["attention_flop_reduction_pct"]
                    flop_red_encoder = flop_info["encoder_flop_reduction_pct"]

                sweep_results.append({
                    "profile_group": group,
                    "eval_group": eval_group_dir.name,
                    "group": eval_group_dir.name,
                    "frac": frac,
                    "flop_reduction": flop_red,
                    "encoder_flop_reduction": flop_red_encoder,
                    "mean_iou": result.get("per_image_mean_iou", 0),
                    "miss_rate": result.get("per_image_miss_rate", 0),
                    "false_pos_rate": result.get("per_image_false_pos_rate", 0),
                    "feature_sim": result.get("mean_feature_cosine_sim", None),
                    "n_frozen": n_frozen,
                })

    return sweep_results


def write_summary_report(sweep_results: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for r in sweep_results:
        grouped[(r["profile_group"], r["eval_group"])].append(r)

    with open(out_path, "w") as f:
        f.write("S-VELCRO SAM ViT-H — Sweep Summary Report\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"{'Profiled On':<14} {'Eval On':<14} {'Frac':<8} {'Frozen':<8} "
                f"{'Attn%':<8} {'Enc%':<8} {'Mean IoU':<10} {'Miss%':<8} {'FP%':<8} {'Feat Sim':<10}\n")
        f.write("-" * 100 + "\n")

        for (prof, evl), entries in sorted(grouped.items()):
            for e in sorted(entries, key=lambda x: x["frac"]):
                feat = f"{e['feature_sim']:.4f}" if e["feature_sim"] is not None else "N/A"
                f.write(f"{prof:<14} {evl:<14} {e['frac']:<8.3f} {e['n_frozen']:<8d} "
                        f"{e['flop_reduction']:<8.1f} {e.get('encoder_flop_reduction', 0):<8.1f} "
                        f"{e['mean_iou']:<10.4f} "
                        f"{e['miss_rate']*100:<8.2f} {e['false_pos_rate']*100:<8.2f} {feat:<10}\n")
            f.write("\n")

    print(f"  Saved summary report to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot SAM VELCRO sweep results")
    parser.add_argument("--results_dir", required=True, help="Path to results_sam_velcro/")
    parser.add_argument("--groups", default="general,animals,vehicles,people",
                        help="Comma-separated groups")
    parser.add_argument("--fractions", default="0.125,0.25,0.375,0.50",
                        help="Comma-separated freeze fractions")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    groups = [g.strip() for g in args.groups.split(",")]
    fractions = [float(f.strip()) for f in args.fractions.split(",")]

    print("[1/7] Loading sweep data...")
    sweep_results = load_sweep_data(results_dir, groups, fractions)
    print(f"  Found {len(sweep_results)} data points")

    if sweep_results:
        print("[2/7] IoU vs FLOP reduction plots...")
        plot_iou_vs_flop_reduction(sweep_results, plots_dir / "iou_vs_flops")

    profile_dir = results_dir / "profile"
    if profile_dir.exists():
        available_groups = [g for g in groups if (profile_dir / g).exists()]

        print("[3/7] Cross-group variance comparison histograms...")
        plot_cross_group_variance_comparison(profile_dir, available_groups, plots_dir)

        print("[4/7] Windowed vs global rank comparison...")
        for g in available_groups:
            plot_windowed_vs_global_rank(
                profile_dir, g,
                plots_dir / f"windowed_vs_global_{g}.png",
            )

        print("[5/7] Before/after variance plots...")
        for frac in fractions:
            manifest_path = None
            for g in groups:
                mp = results_dir / f"freeze_{g}_f{frac}" / "freeze_manifest.json"
                if mp.exists():
                    manifest_path = mp
                    break
            if manifest_path:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                plot_before_after_variance(
                    profile_dir, available_groups,
                    manifest.get("frozen_heads", []),
                    plots_dir / f"before_after_f{frac}",
                )

    print("[6/7] Domain specialization matrix...")
    matrix_data = {}
    for r in sweep_results:
        key = (r["profile_group"], r["eval_group"])
        if key not in matrix_data or r["frac"] == 0.25:
            matrix_data[key] = r["mean_iou"]
    if matrix_data:
        all_groups_in_results = sorted(set(
            g for pair in matrix_data.keys() for g in pair
        ))
        plot_domain_specialization_matrix(
            matrix_data, all_groups_in_results,
            plots_dir / "domain_specialization_matrix.png",
        )

    print("[7/7] Summary report...")
    write_summary_report(sweep_results, plots_dir / "summary_report.txt")

    print("\nDone! All plots saved to:", plots_dir)


if __name__ == "__main__":
    main()
