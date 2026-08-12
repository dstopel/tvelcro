#!/usr/bin/env python3
"""
plot_accuracy_vs_flops.py

Reads DINOv2 ViT-Giant S-VELCRO sweep results and generates:
  - Per-group accuracy vs FLOP reduction PNGs (top-1 + top-5)
  - Combined all-groups top-1 PNG
  - Summary report (text table)
  - Per-group frozen-heads reports
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUM_BLOCKS = 40
NUM_HEADS = 24
GROUPS = ["dogs", "cats", "cars", "snakes", "fish", "boats", "birds"]
GROUP_COLORS = {
    "dogs": "tab:blue",
    "cats": "tab:orange",
    "cars": "tab:green",
    "snakes": "tab:red",
    "fish": "tab:purple",
    "boats": "tab:brown",
    "birds": "tab:cyan",
}


def load_sweep_data(results_dir: Path):
    data = defaultdict(list)

    for frac_dir in sorted(results_dir.iterdir()):
        if not frac_dir.is_dir() or not frac_dir.name.startswith("frac"):
            continue
        frac_str = frac_dir.name[4:]

        for group in GROUPS:
            ri_path = frac_dir / f"eval_{group}" / "meta" / "run_info.json"
            fm_path = frac_dir / f"freeze_{group}" / "freeze_manifest.json"

            if not ri_path.exists():
                print(f"  [WARN] missing {ri_path}")
                continue

            with open(ri_path) as f:
                ri = json.load(f)
            fm = None
            if fm_path.exists():
                with open(fm_path) as f:
                    fm = json.load(f)

            acc = ri.get("accuracy", {}).get(group, {})
            flop_red = ri.get("flop_savings_pct", 0.0)

            entry = {
                "frac": frac_str,
                "frac_float": float(frac_str),
                "flop_reduction": flop_red,
                "top1": acc.get("top1", 0.0),
                "top5": acc.get("top5", 0.0),
                "total": acc.get("total", 0),
                "manifest": fm,
            }
            data[group].append(entry)

    for group in data:
        data[group].sort(key=lambda x: x["flop_reduction"])

    return data


def plot_per_group(data, plots_dir: Path):
    for group in GROUPS:
        entries = data.get(group, [])
        if not entries:
            continue

        x = [e["flop_reduction"] for e in entries]
        y1 = [e["top1"] for e in entries]
        y5 = [e["top5"] for e in entries]
        fracs = [e["frac"] for e in entries]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y1, "o-", color="tab:blue", label="Top-1", markersize=7)
        ax.plot(x, y5, "o--", color="tab:orange", label="Top-5", markersize=7)

        for xi, y1i, y5i, fr in zip(x, y1, y5, fracs):
            ax.annotate(fr, (xi, y1i), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center", color="tab:blue")
            ax.annotate(fr, (xi, y5i), textcoords="offset points",
                        xytext=(0, -14), fontsize=7, ha="center", color="tab:orange")

        ax.set_xlabel("Attention FLOP Reduction (%)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"S-VELCRO DINOv2 ViT-Giant — {group.capitalize()}: Accuracy vs FLOP Reduction",
                     fontsize=13)
        ax.set_xlim(-2, 87)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = plots_dir / f"{group}_accuracy_vs_flops.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_combined_top1(data, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for group in GROUPS:
        entries = data.get(group, [])
        if not entries:
            continue
        x = [e["flop_reduction"] for e in entries]
        y1 = [e["top1"] for e in entries]
        ax.plot(x, y1, "o-", color=GROUP_COLORS[group], label=group.capitalize(), markersize=7)

    ax.set_xlabel("Attention FLOP Reduction (%)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("S-VELCRO DINOv2 ViT-Giant — All Groups: Top-1 Accuracy vs FLOP Reduction",
                 fontsize=13)
    ax.set_xlim(-2, 87)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = plots_dir / "all_groups_top1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_summary_report(data, plots_dir: Path):
    all_fracs = sorted({e["frac_float"] for g in data.values() for e in g})

    idx = {}
    for group, entries in data.items():
        for e in entries:
            idx[(group, e["frac_float"])] = e

    out_path = plots_dir / "summary_report.txt"
    with open(out_path, "w") as f:
        hdr = f"{'FRAC':<8} {'FLOP Red%':<12}"
        for g in GROUPS:
            hdr += f" {g+' top1':<12} {g+' top5':<12}"
        f.write(hdr + "\n")
        f.write("=" * len(hdr) + "\n")

        for frac in all_fracs:
            flop_red = "-"
            for g in GROUPS:
                if (g, frac) in idx:
                    flop_red = f"{idx[(g, frac)]['flop_reduction']:.2f}"
                    break

            line = f"{frac:<8.2f} {flop_red:<12}"
            for g in GROUPS:
                if (g, frac) in idx:
                    e = idx[(g, frac)]
                    line += f" {e['top1']:<12.2f} {e['top5']:<12.2f}"
                else:
                    line += f" {'N/A':<12} {'N/A':<12}"
            f.write(line + "\n")

    print(f"  Saved {out_path}")


def write_frozen_heads_reports(data, plots_dir: Path):
    for group in GROUPS:
        entries = data.get(group, [])
        if not entries:
            continue

        out_path = plots_dir / f"{group}_frozen_heads_report.txt"
        with open(out_path, "w") as f:
            f.write(f"S-VELCRO DINOv2 ViT-Giant — {group.capitalize()} Frozen Heads Report\n")
            f.write("=" * 70 + "\n\n")

            for e in entries:
                frac = e["frac"]
                flop_red = e["flop_reduction"]
                top1 = e["top1"]
                top5 = e["top5"]
                manifest = e["manifest"]

                f.write(f"=== FRAC={frac} (FLOP Reduction: {flop_red:.1f}%) ===\n")

                if manifest is None:
                    f.write("  [no freeze manifest found]\n")
                else:
                    frozen_heads_list = manifest.get("frozen_heads", [])

                    block_frozen = defaultdict(list)
                    block_ranks = defaultdict(list)
                    for fh in frozen_heads_list:
                        b = fh["block"]
                        h = fh["head"]
                        block_frozen[b].append(h)
                        if "channel_rank" in fh:
                            block_ranks[b].append((h, fh["channel_rank"]))

                    all_heads = set(range(NUM_HEADS))
                    for b in range(NUM_BLOCKS):
                        frozen = sorted(block_frozen.get(b, []))
                        active = sorted(all_heads - set(frozen))
                        f.write(f"  Block {b:<3} Frozen: {str(frozen):<40} Active: {active}\n")
                        if b in block_ranks:
                            ranks_str = ", ".join(
                                f"h{h}={r:.4f}" for h, r in sorted(block_ranks[b])
                            )
                            f.write(f"            Channel ranks: {ranks_str}\n")

                f.write(f"  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%\n\n")

        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs FLOP reduction for DINOv2 ViT-Giant S-VELCRO sweep"
    )
    parser.add_argument("--results_dir", required=True, help="Path to results_vit_giant_sweep/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading sweep data...")
    data = load_sweep_data(results_dir)
    for g in GROUPS:
        print(f"  {g}: {len(data.get(g, []))} data points")

    print("[2/5] Per-group accuracy plots...")
    plot_per_group(data, plots_dir)

    print("[3/5] Combined top-1 plot...")
    plot_combined_top1(data, plots_dir)

    print("[4/5] Summary report...")
    write_summary_report(data, plots_dir)

    print("[5/5] Frozen heads reports...")
    write_frozen_heads_reports(data, plots_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
