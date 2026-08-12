#!/usr/bin/env python3
"""
plot_results.py — accuracy vs FLOP-reduction plots for T-VELCRO sweep results.

Generalizes _legacy_giant/plot_accuracy_vs_flops.py to both models and to
multi-seed aggregation (mean line plus min-max band when several seeds exist).

Expected layout under --results_dir (produced by experiments/vit/*.sh):
  [seed<S>/]                    optional seed level; absent = single run
    tk<K>/ or frac<F>/          one operating point
      eval_<label>/meta/run_info.json

Series are discovered from eval_* dir names; groups/domains are not hardcoded.
Outputs into <results_dir>/plots/:
  all_series_top1.png, <label>_accuracy.png per series, summary_report.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito-derived categorical palette, CVD-validated (dataviz six checks).
# Fixed order, assigned to sorted series labels -- color follows the entity.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#B0A000", "#7B5EA7"]


def discover(results_dir: Path):
    """Returns rows [(seed, op, label, flop_red, top1, top5)] and a model name."""
    seed_dirs = sorted(d for d in results_dir.iterdir()
                       if d.is_dir() and d.name.startswith("seed"))
    if not seed_dirs:
        seed_dirs = [results_dir]

    rows = []
    model_name = None
    for sd in seed_dirs:
        seed = sd.name if sd is not results_dir else "seed*"
        for op in sorted(d for d in sd.iterdir() if d.is_dir()
                         and (d.name.startswith("tk") or d.name.startswith("frac"))):
            for ev in sorted(op.glob("eval_*")):
                ri_path = ev / "meta" / "run_info.json"
                if not ri_path.exists():
                    print(f"  [WARN] missing {ri_path}")
                    continue
                with open(ri_path) as f:
                    ri = json.load(f)
                model_name = ri.get("model", model_name)
                suffix = ev.name[len("eval_"):]
                acc = ri.get("accuracy", {})
                for g, a in acc.items():
                    label = suffix if len(acc) == 1 else f"{suffix}:{g}"
                    rows.append((seed, op.name, label,
                                 float(ri.get("flop_savings_pct", 0.0)),
                                 float(a.get("top1", 0.0)),
                                 float(a.get("top5", 0.0))))
    return rows, model_name


def aggregate(rows):
    """{label: [(x_mean, dict(metric -> (mean, min, max)), n_seeds), ...] sorted by x}"""
    by_key = defaultdict(list)
    for seed, op, label, x, t1, t5 in rows:
        by_key[(label, op)].append((x, t1, t5))

    series = defaultdict(list)
    for (label, op), vals in by_key.items():
        arr = np.array(vals)  # (n_seeds, 3)
        series[label].append({
            "op": op,
            "n_seeds": len(vals),
            "x": float(arr[:, 0].mean()),
            "top1": (float(arr[:, 1].mean()), float(arr[:, 1].min()), float(arr[:, 1].max())),
            "top5": (float(arr[:, 2].mean()), float(arr[:, 2].min()), float(arr[:, 2].max())),
        })
    for label in series:
        series[label].sort(key=lambda e: e["x"])
    return dict(series)


def color_for(label, labels):
    i = sorted(labels).index(label)
    if i >= len(PALETTE):
        print(f"  [WARN] more than {len(PALETTE)} series; "
              f"'{label}' reuses a hue with a dashed line")
    return PALETTE[i % len(PALETTE)], "-" if i < len(PALETTE) else "--"


def _line_with_band(ax, entries, metric, color, label, linestyle="-"):
    x = [e["x"] for e in entries]
    mean = [e[metric][0] for e in entries]
    lo = [e[metric][1] for e in entries]
    hi = [e[metric][2] for e in entries]
    multi_seed = any(e["n_seeds"] > 1 for e in entries)
    ax.plot(x, mean, marker="o", markersize=7, linewidth=2, linestyle=linestyle,
            color=color, label=label, markeredgecolor="white", markeredgewidth=1)
    if multi_seed:
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)


def _style_axes(ax, title, ylabel="Top-1 Accuracy (%)"):
    ax.set_xlabel("Attention FLOP Reduction (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_combined(series, model_name, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(series)
    for label in sorted(labels):
        color, ls = color_for(label, labels)
        _line_with_band(ax, series[label], "top1", color, label, ls)
    _style_axes(ax, f"T-VELCRO {model_name or ''} — Top-1 Accuracy vs FLOP Reduction")
    if len(labels) > 1:
        ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    out = plots_dir / "all_series_top1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_per_series(series, model_name, plots_dir: Path):
    for label, entries in sorted(series.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        _line_with_band(ax, entries, "top1", PALETTE[0], "Top-1")
        _line_with_band(ax, entries, "top5", PALETTE[1], "Top-5", linestyle="--")
        # Selective direct labels: operating point on the top-1 endpoint marks
        for e in entries:
            ax.annotate(e["op"], (e["x"], e["top1"][0]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, ha="center", color="#444444")
        _style_axes(ax, f"T-VELCRO {model_name or ''} — {label}: Accuracy vs "
                        f"FLOP Reduction", ylabel="Accuracy (%)")
        ax.legend(fontsize=10, frameon=False)
        fig.tight_layout()
        out = plots_dir / f"{label.replace(':', '_')}_accuracy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def write_summary(series, model_name, plots_dir: Path):
    out = plots_dir / "summary_report.txt"
    with open(out, "w") as f:
        f.write(f"T-VELCRO sweep summary — model: {model_name or 'unknown'}\n")
        f.write(f"{'series':<24} {'op':<10} {'flop_red%':<10} {'seeds':<6} "
                f"{'top1_mean':<10} {'top1_min':<10} {'top1_max':<10} "
                f"{'top5_mean':<10}\n")
        f.write("-" * 92 + "\n")
        for label in sorted(series):
            for e in series[label]:
                f.write(f"{label:<24} {e['op']:<10} {e['x']:<10.2f} "
                        f"{e['n_seeds']:<6} {e['top1'][0]:<10.2f} "
                        f"{e['top1'][1]:<10.2f} {e['top1'][2]:<10.2f} "
                        f"{e['top5'][0]:<10.2f}\n")
    print(f"  Saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot T-VELCRO accuracy vs FLOP "
                                             "reduction (multi-seed aware)")
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows, model_name = discover(results_dir)
    if not rows:
        raise SystemExit(f"No eval_*/meta/run_info.json found under {results_dir}")

    n_seeds = len({r[0] for r in rows})
    series = aggregate(rows)
    print(f"Found {len(rows)} data points, {len(series)} series, {n_seeds} seed(s)")

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_combined(series, model_name, plots_dir)
    plot_per_series(series, model_name, plots_dir)
    write_summary(series, model_name, plots_dir)


if __name__ == "__main__":
    main()
