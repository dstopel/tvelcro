#!/usr/bin/env python3
"""
Generate locality comparison plots showing general variance > domain variance.
Combines profiling data from results_cross_domain and results_wave2.
Produces a single multi-panel figure for the thesis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────

PROFILE_DIRS = [
    Path("results_cross_domain/profile"),  # general, kvasir, isic
    Path("results_wave2/profile"),         # general (symlink), busi, pcb, deepglobe
]

DOMAINS = ["general", "kvasir", "isic", "busi", "pcb", "deepglobe"]
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

OUT_DIR = Path("plots_locality_comparison")
OUT_DIR.mkdir(exist_ok=True)


# ── Load profile data ────────────────────────────────────────────────────────

def load_rank_scores(profile_dir, group):
    """Load all head variance .npy files and compute rank scores."""
    group_dir = profile_dir / group
    if not group_dir.exists():
        return None

    scores = []
    for block in range(NUM_BLOCKS):
        for head in range(NUM_HEADS):
            var_file = group_dir / f"block{block}_head{head}_variance.npy"
            if var_file.exists():
                var = np.load(var_file)
                scores.append({
                    "block": block,
                    "head": head,
                    "score": float(var.sum()),
                    "type": "global" if block in GLOBAL_BLOCKS else "windowed",
                })
    return scores if len(scores) == 512 else None


# Find profile data for each domain
domain_scores = {}
for domain in DOMAINS:
    for pdir in PROFILE_DIRS:
        scores = load_rank_scores(pdir, domain)
        if scores:
            domain_scores[domain] = scores
            break

print(f"Loaded profiles for: {list(domain_scores.keys())}")
if "general" not in domain_scores:
    print("ERROR: general profile not found!")
    exit(1)


# ── Figure 1: Windowed head variance — General vs All Domains ────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Value Locality: Windowed Head Variance — General vs Specialized Domains",
             fontsize=14, fontweight="bold", y=0.98)

specialized = [d for d in DOMAINS if d != "general" and d in domain_scores]
general_windowed = [s["score"] for s in domain_scores["general"] if s["type"] == "windowed"]

for idx, domain in enumerate(specialized):
    ax = axes[idx // 3, idx % 3]
    domain_windowed = [s["score"] for s in domain_scores[domain] if s["type"] == "windowed"]

    bins = np.linspace(0, max(np.percentile(general_windowed, 99), np.percentile(domain_windowed, 99)), 50)

    ax.hist(general_windowed, bins=bins, alpha=0.6, color="red", label=f"General (median={np.median(general_windowed):.0f})")
    ax.hist(domain_windowed, bins=bins, alpha=0.6, color="blue", label=f"{DOMAIN_LABELS[domain]} (median={np.median(domain_windowed):.0f})")

    ax.axvline(np.median(general_windowed), color="red", linestyle="--", linewidth=1.5)
    ax.axvline(np.median(domain_windowed), color="blue", linestyle="--", linewidth=1.5)

    ax.set_title(DOMAIN_LABELS[domain], fontsize=11)
    ax.set_xlabel("Rank Score (Σ variance)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

# Hide unused subplot if < 5 domains
if len(specialized) < 6:
    axes[1, 2].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / "windowed_variance_general_vs_domains.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "windowed_variance_general_vs_domains.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR}/windowed_variance_general_vs_domains.png")
plt.close()


# ── Figure 2: Global head variance — General vs All Domains ──────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Value Locality: Global Head Variance — General vs Specialized Domains",
             fontsize=14, fontweight="bold", y=0.98)

general_global = [s["score"] for s in domain_scores["general"] if s["type"] == "global"]

for idx, domain in enumerate(specialized):
    ax = axes[idx // 3, idx % 3]
    domain_global = [s["score"] for s in domain_scores[domain] if s["type"] == "global"]

    bins = np.linspace(0, max(np.percentile(general_global, 99), np.percentile(domain_global, 99)), 30)

    ax.hist(general_global, bins=bins, alpha=0.6, color="red", label=f"General (median={np.median(general_global):.0f})")
    ax.hist(domain_global, bins=bins, alpha=0.6, color="blue", label=f"{DOMAIN_LABELS[domain]} (median={np.median(domain_global):.0f})")

    ax.axvline(np.median(general_global), color="red", linestyle="--", linewidth=1.5)
    ax.axvline(np.median(domain_global), color="blue", linestyle="--", linewidth=1.5)

    ax.set_title(DOMAIN_LABELS[domain], fontsize=11)
    ax.set_xlabel("Rank Score (Σ variance)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

if len(specialized) < 6:
    axes[1, 2].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / "global_variance_general_vs_domains.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "global_variance_general_vs_domains.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR}/global_variance_general_vs_domains.png")
plt.close()


# ── Figure 3: Per-block mean variance — All domains on one plot ──────────────

fig, ax = plt.subplots(figsize=(14, 6))

colors = {"general": "black", "kvasir": "green", "isic": "orange",
          "busi": "purple", "pcb": "red", "deepglobe": "blue"}

for domain in ["general"] + specialized:
    scores = domain_scores[domain]
    block_means = []
    for block in range(NUM_BLOCKS):
        block_scores = [s["score"] for s in scores if s["block"] == block]
        block_means.append(np.mean(block_scores))

    style = "-" if domain == "general" else "--"
    lw = 2.5 if domain == "general" else 1.5
    ax.plot(range(NUM_BLOCKS), block_means, style, linewidth=lw,
            color=colors.get(domain, "gray"), label=DOMAIN_LABELS.get(domain, domain))

# Mark global blocks
for gb in GLOBAL_BLOCKS:
    ax.axvline(gb, color="gray", alpha=0.3, linestyle=":")

ax.set_xlabel("Block Index", fontsize=12)
ax.set_ylabel("Mean Rank Score (Σ variance)", fontsize=12)
ax.set_title("Per-Block Mean Variance — General vs Specialized Domains", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_xticks(range(0, 32, 2))
ax.annotate("Global blocks", xy=(23, ax.get_ylim()[1] * 0.9), fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(OUT_DIR / "per_block_variance_all_domains.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "per_block_variance_all_domains.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR}/per_block_variance_all_domains.png")
plt.close()


# ── Figure 4: Summary bar chart — Total variance per domain ──────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

all_domains_ordered = ["general"] + sorted(specialized, key=lambda d: sum(s["score"] for s in domain_scores[d] if s["type"] == "windowed"))

# Windowed
windowed_means = [np.mean([s["score"] for s in domain_scores[d] if s["type"] == "windowed"]) for d in all_domains_ordered]
bar_colors = [colors.get(d, "gray") for d in all_domains_ordered]
bars = ax1.bar([DOMAIN_LABELS.get(d, d) for d in all_domains_ordered], windowed_means, color=bar_colors, alpha=0.7, edgecolor="black")
ax1.set_ylabel("Mean Rank Score")
ax1.set_title("Windowed Heads — Mean Variance per Domain", fontweight="bold")
ax1.tick_params(axis="x", rotation=25)
ax1.axhline(windowed_means[0], color="black", linestyle="--", alpha=0.5, label="General baseline")
ax1.legend()

# Global
all_domains_ordered_g = ["general"] + sorted(specialized, key=lambda d: sum(s["score"] for s in domain_scores[d] if s["type"] == "global"))
global_means = [np.mean([s["score"] for s in domain_scores[d] if s["type"] == "global"]) for d in all_domains_ordered_g]
bar_colors_g = [colors.get(d, "gray") for d in all_domains_ordered_g]
bars = ax2.bar([DOMAIN_LABELS.get(d, d) for d in all_domains_ordered_g], global_means, color=bar_colors_g, alpha=0.7, edgecolor="black")
ax2.set_ylabel("Mean Rank Score")
ax2.set_title("Global Heads — Mean Variance per Domain", fontweight="bold")
ax2.tick_params(axis="x", rotation=25)
ax2.axhline(global_means[0], color="black", linestyle="--", alpha=0.5, label="General baseline")
ax2.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "variance_bar_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "variance_bar_comparison.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR}/variance_bar_comparison.png")
plt.close()


# ── Figure 5: Windowed vs Global split per domain ────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Windowed vs Global Head Rank Scores per Domain",
             fontsize=14, fontweight="bold", y=0.98)

all_plot_domains = ["general"] + specialized
for idx, domain in enumerate(all_plot_domains[:6]):
    ax = axes[idx // 3, idx % 3]
    windowed = [s["score"] for s in domain_scores[domain] if s["type"] == "windowed"]
    global_ = [s["score"] for s in domain_scores[domain] if s["type"] == "global"]

    max_val = np.percentile(global_, 95)
    bins_w = np.linspace(0, min(np.percentile(windowed, 99), max_val), 40)
    bins_g = np.linspace(0, max_val, 20)

    ax.hist(windowed, bins=bins_w, alpha=0.7, color="steelblue", label=f"Windowed (n=448, med={np.median(windowed):.0f})")
    ax.hist(global_, bins=bins_g, alpha=0.7, color="indianred", label=f"Global (n=64, med={np.median(global_):.0f})")

    ax.axvline(np.median(windowed), color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(np.median(global_), color="indianred", linestyle="--", linewidth=1.5)

    ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=11)
    ax.set_xlabel("Rank Score (Σ variance)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / "windowed_vs_global_all_domains.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT_DIR / "windowed_vs_global_all_domains.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR}/windowed_vs_global_all_domains.png")
plt.close()


# ── Print summary ────────────────────────────────────────────────────────────

print()
print("=" * 80)
print("LOCALITY EVIDENCE: General variance vs Domain-specific variance")
print("=" * 80)
print()
print(f"{'Domain':<20} {'Windowed Mean':>15} {'vs General':>12} {'Global Mean':>15} {'vs General':>12}")
print("-" * 80)

gen_w = np.mean([s["score"] for s in domain_scores["general"] if s["type"] == "windowed"])
gen_g = np.mean([s["score"] for s in domain_scores["general"] if s["type"] == "global"])

for domain in ["general"] + specialized:
    w_mean = np.mean([s["score"] for s in domain_scores[domain] if s["type"] == "windowed"])
    g_mean = np.mean([s["score"] for s in domain_scores[domain] if s["type"] == "global"])

    w_pct = ((w_mean - gen_w) / gen_w) * 100 if domain != "general" else 0
    g_pct = ((g_mean - gen_g) / gen_g) * 100 if domain != "general" else 0

    w_indicator = f"{w_pct:+.1f}%" if domain != "general" else "baseline"
    g_indicator = f"{g_pct:+.1f}%" if domain != "general" else "baseline"

    print(f"{DOMAIN_LABELS.get(domain, domain):<20} {w_mean:>15.1f} {w_indicator:>12} {g_mean:>15.1f} {g_indicator:>12}")

print()
print("Negative % = MORE locality than general (lower variance = more freezable)")
print("=" * 80)
print()
print(f"All plots saved to: {OUT_DIR}/")
