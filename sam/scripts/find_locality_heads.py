#!/usr/bin/env python3
"""
Find heads with strongest locality and plot them in the same style as
plot_variance_distributions.py comparison/heads plots:
  - Histogram of per-dimension variance values
  - histtype='step' (line histograms overlaid)
  - One line per group (general + domain)
  - Grid, legend, same axis labels
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROFILE_DIRS = [
    Path("results_cross_domain/profile"),
    Path("results_wave2/profile"),
]

DOMAINS = ["kvasir", "isic", "busi", "pcb", "deepglobe"]
DOMAIN_LABELS = {
    "general": "General (COCO)",
    "kvasir": "Kvasir (Endoscopy)",
    "isic": "ISIC (Dermoscopy)",
    "busi": "BUSI (Ultrasound)",
    "pcb": "PCB (Circuit Board)",
    "deepglobe": "DeepGlobe (Satellite)",
}

GLOBAL_BLOCKS = {7, 15, 23, 31}
NUM_BLOCKS = 32
NUM_HEADS = 16
BINS = 60
TOP_N = 6

OUT_DIR = Path("plots_locality_heads")
OUT_DIR.mkdir(exist_ok=True)


# ── Load data ────────────────────────────────────────────────────────────────

def find_profile_dir(group):
    for pdir in PROFILE_DIRS:
        gdir = pdir / group
        if gdir.exists() and len(list(gdir.glob("*.npy"))) > 500:
            return gdir
    return None


def load_variance_per_head(profile_dir):
    data = {}
    for block in range(NUM_BLOCKS):
        for head in range(NUM_HEADS):
            var_file = profile_dir / f"block{block}_head{head}_variance.npy"
            if var_file.exists():
                data[(block, head)] = np.load(var_file)
    return data


general_dir = find_profile_dir("general")
if not general_dir:
    print("ERROR: general profile not found")
    exit(1)

general_data = load_variance_per_head(general_dir)
print(f"Loaded general: {len(general_data)} heads")

domain_data = {}
for domain in DOMAINS:
    ddir = find_profile_dir(domain)
    if ddir:
        domain_data[domain] = load_variance_per_head(ddir)
        print(f"Loaded {domain}: {len(domain_data[domain])} heads")


# ── Find heads with strongest locality ───────────────────────────────────────

print()
print("=" * 90)
print("HEADS WITH STRONGEST LOCALITY (domain variance << general variance)")
print("=" * 90)

domain_top_heads = {}

for domain in DOMAINS:
    if domain not in domain_data:
        continue

    ratios = []
    for (block, head), gen_var in general_data.items():
        if (block, head) in domain_data[domain]:
            dom_var = domain_data[domain][(block, head)]
            gen_score = gen_var.sum()
            dom_score = dom_var.sum()
            if gen_score < 1e-6:
                continue
            ratio = dom_score / gen_score
            block_type = "GLOBAL" if block in GLOBAL_BLOCKS else "windowed"
            ratios.append({
                "block": block, "head": head, "ratio": ratio,
                "gen_score": gen_score, "dom_score": dom_score,
                "type": block_type,
            })

    ratios.sort(key=lambda x: x["ratio"])
    domain_top_heads[domain] = ratios[:TOP_N]

    print(f"\n  {DOMAIN_LABELS[domain]} — Top {TOP_N} heads with most locality:")
    print(f"  {'Block':<7} {'Head':<6} {'Type':<10} {'General Σvar':>12} {domain+' Σvar':>12} {'Ratio':>8} {'Variance reduction':>20}")
    print(f"  {'-'*75}")
    for r in ratios[:TOP_N]:
        reduction = (1.0 - r["ratio"]) * 100
        print(f"  {r['block']:<7} {r['head']:<6} {r['type']:<10} {r['gen_score']:>12.1f} {r['dom_score']:>12.1f} {r['ratio']:>8.3f} {reduction:>19.1f}%")


# ── Plot in comparison/heads style ───────────────────────────────────────────

print()
print("Generating plots (same style as comparison/heads)...")

for domain in DOMAINS:
    if domain not in domain_top_heads:
        continue

    top_heads = domain_top_heads[domain]
    n_plots = len(top_heads)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.suptitle(f"Locality Evidence: {DOMAIN_LABELS[domain]} vs General\n"
                 f"Per-dimension variance histograms — domain (blue) has lower variance than general (red)",
                 fontsize=13, fontweight="bold", y=0.99)

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for idx, head_info in enumerate(top_heads):
        ax = axes_flat[idx]
        block = head_info["block"]
        head = head_info["head"]
        block_type = head_info["type"]

        gen_var = general_data[(block, head)].ravel()
        dom_var = domain_data[domain][(block, head)].ravel()

        # Same style as plot_variance_distributions.py
        all_vals = np.concatenate([gen_var, dom_var])
        p99 = np.percentile(all_vals, 99) if len(all_vals) > 0 else 1.0

        ax.hist(gen_var, bins=BINS, range=(0, p99), histtype='step',
                label=f"general (Σ={gen_var.sum():.0f})", linewidth=2, color='red')
        ax.hist(dom_var, bins=BINS, range=(0, p99), histtype='step',
                label=f"{domain} (Σ={dom_var.sum():.0f})", linewidth=2, color='blue')

        reduction = (1.0 - head_info["ratio"]) * 100
        ax.set_title(f"Block {block} Head {head} [{block_type}] — variance reduction: {reduction:.1f}%",
                     fontsize=10)
        ax.set_xlabel("Variance")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for idx in range(len(top_heads), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT_DIR / f"locality_heads_{domain}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_DIR / f"locality_heads_{domain}.pdf", bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


# ── Single best head per domain (one combined figure) ────────────────────────

n_domains = len([d for d in DOMAINS if d in domain_top_heads])
fig, axes = plt.subplots(1, n_domains, figsize=(7 * n_domains, 5))
if n_domains == 1:
    axes = [axes]

fig.suptitle("Best Locality Head per Domain — Variance Distribution Comparison",
             fontsize=13, fontweight="bold")

for idx, domain in enumerate(d for d in DOMAINS if d in domain_top_heads):
    ax = axes[idx]
    best = domain_top_heads[domain][0]
    block, head = best["block"], best["head"]
    block_type = best["type"]

    gen_var = general_data[(block, head)].ravel()
    dom_var = domain_data[domain][(block, head)].ravel()

    all_vals = np.concatenate([gen_var, dom_var])
    p99 = np.percentile(all_vals, 99) if len(all_vals) > 0 else 1.0

    ax.hist(gen_var, bins=BINS, range=(0, p99), histtype='step',
            label=f"general (Σ={gen_var.sum():.0f})", linewidth=2, color='red')
    ax.hist(dom_var, bins=BINS, range=(0, p99), histtype='step',
            label=f"{domain} (Σ={dom_var.sum():.0f})", linewidth=2, color='blue')

    reduction = (1.0 - best["ratio"]) * 100
    ax.set_title(f"{DOMAIN_LABELS[domain]}\nBlock {block} Head {head} [{block_type}] — −{reduction:.0f}%",
                 fontsize=10)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "best_locality_head_per_domain.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "best_locality_head_per_domain.pdf", bbox_inches="tight")
print(f"  Saved: {OUT_DIR}/best_locality_head_per_domain.png")
plt.close()

print(f"\nAll plots saved to: {OUT_DIR}/")
