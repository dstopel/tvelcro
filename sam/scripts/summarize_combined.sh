#!/usr/bin/env bash
# summarize_combined.sh — Compute ACTUAL FLOP savings from real manifests + profiles
#
# Usage:
#   bash scripts/summarize_combined.sh
#   OUT_DIR=./results_combined_freeze bash scripts/summarize_combined.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-./results_combined_freeze}"

if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUT_DIR"
    exit 1
fi

python3 - "$OUT_DIR" <<'PYEOF'
import json, os, glob, sys
from pathlib import Path
sys.path.insert(0, '.')

from sam_velcro_pkg.flops import (
    compute_flop_reduction,
    compute_mlp_flop_reduction,
    TOTAL_ENCODER_FLOPS,
    TOTAL_ATTN_FLOPS,
    TOTAL_MLP_FLOPS,
    GLOBAL_ATTN_INDEXES,
    PER_HEAD_SAVINGS_WINDOWED,
    PER_HEAD_SAVINGS_GLOBAL,
    PER_NEURON_SAVINGS_WINDOWED,
    PER_NEURON_SAVINGS_GLOBAL,
)
from sam_velcro_pkg.mlp_hooks import select_mlp_neurons_to_freeze
from sam_velcro_pkg.constants import NUM_BLOCKS, MLP_DIM

out = sys.argv[1]

# Find attention manifest per domain
def find_attn_manifest(domain):
    candidates = [
        f'./results_cross_domain/{domain}/freeze_{domain}_f0.25/freeze_manifest.json',
        f'./results_cross_domain/freeze_{domain}_f0.25/freeze_manifest.json',
        f'./results_wave2/{domain}/freeze_{domain}_f0.25/freeze_manifest.json',
        f'./results_wave2/freeze_{domain}_f0.25/freeze_manifest.json',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


print()
print('=' * 135)
print('  Combined Freeze — ACTUAL FLOP Savings (per-domain manifests + MLP profiles)')
print('=' * 135)
print()
print(f"  Encoder: {TOTAL_ENCODER_FLOPS/1e9:.1f} GMACs | Attn: {TOTAL_ATTN_FLOPS/1e9:.1f} GMACs ({TOTAL_ATTN_FLOPS/TOTAL_ENCODER_FLOPS*100:.1f}%) | MLP: {TOTAL_MLP_FLOPS/1e9:.1f} GMACs ({TOTAL_MLP_FLOPS/TOTAL_ENCODER_FLOPS*100:.1f}%)")
print(f"  Per head: windowed={PER_HEAD_SAVINGS_WINDOWED/1e6:.1f}M, global={PER_HEAD_SAVINGS_GLOBAL/1e6:.1f}M MACs")
print(f"  Per neuron: windowed={PER_NEURON_SAVINGS_WINDOWED/1e6:.2f}M, global={PER_NEURON_SAVINGS_GLOBAL/1e6:.2f}M MACs")
print()
print(f"{'Domain':<8} {'MLP%':<6} {'IoU':<7} {'Miss%':<7} {'Dom':<5} {'BG':<6} {'Cov':<5} {'AttnH':<8} {'AttnSave':<10} {'MLPnrns':<10} {'MLPSave':<10} {'TOTAL':<8} {'GMACs':<8}")
print('-' * 135)

for d in ['kvasir', 'isic', 'busi', 'pcb']:
    results = sorted(glob.glob(os.path.join(out, d, 'combined_f*', 'results.json')))
    if not results:
        print(f'{d:<8} -- no results found')
        continue

    # Load ACTUAL attention manifest for this domain
    attn_manifest_path = find_attn_manifest(d)
    if attn_manifest_path:
        with open(attn_manifest_path) as f:
            attn_manifest = json.load(f)
        frozen_heads_list = attn_manifest.get('frozen_heads', [])
        attn_flop = compute_flop_reduction(frozen_heads_list)
        attn_save_pct = attn_flop['encoder_flop_reduction_pct']
        attn_save_gmacs = attn_flop['saved_gflops']
        n_wind_h = attn_flop['windowed_heads_frozen']
        n_glob_h = attn_flop['global_heads_frozen']
        attn_info = f"{n_wind_h}w+{n_glob_h}g"
    else:
        attn_save_pct = 0
        attn_save_gmacs = 0
        attn_info = "??"

    # MLP profile directory for this domain
    mlp_profile_dir = Path(out) / d / 'mlp_profile'

    for jf in results:
        try:
            r = json.load(open(jf))
            mlp_frac = r['mlp_freeze_fraction']
            iou = r['mask_comparison']['mean_iou']
            miss = r['mask_comparison']['miss_rate']
            dom = r['miss_decomposition']['domain_misses']
            bg = r['miss_decomposition']['background_misses']
            frz_c = r['coverage_comparison']['frozen_mean_coverage']

            # Compute ACTUAL MLP FLOP savings by re-selecting neurons from profiles
            if mlp_profile_dir.exists():
                mlp_frozen = select_mlp_neurons_to_freeze(
                    mlp_profile_dir, d, mlp_frac
                )
                # Count per block
                neurons_per_block = {}
                for (block_idx, neuron_idx) in mlp_frozen.keys():
                    neurons_per_block[block_idx] = neurons_per_block.get(block_idx, 0) + 1

                mlp_flop = compute_mlp_flop_reduction(neurons_per_block)
                mlp_save_pct = mlp_flop['encoder_flop_reduction_pct']
                mlp_save_gmacs = mlp_flop['saved_gflops']
                n_wind_n = mlp_flop['windowed_neurons_frozen']
                n_glob_n = mlp_flop['global_neurons_frozen']
                mlp_info = f"{n_wind_n}w+{n_glob_n}g"
            else:
                mlp_save_pct = 0
                mlp_save_gmacs = 0
                mlp_info = "??"

            total_save_pct = attn_save_pct + mlp_save_pct
            total_save_gmacs = attn_save_gmacs + mlp_save_gmacs

            print(f'{d:<8} {mlp_frac*100:>5.1f}% {iou:<7.4f} {miss*100:<6.1f}% {dom:<5} {bg:<6} {frz_c*100:<4.0f}% {attn_info:<8} {attn_save_pct:>5.1f}%     {mlp_info:<10} {mlp_save_pct:>5.1f}%     {total_save_pct:>5.1f}%  {total_save_gmacs:>7.1f}')
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f'{d:<8} ERROR: {e}')

    if attn_manifest_path:
        print(f'         attn src: {attn_manifest_path}')
    print()

print('=' * 135)

# Aggregates
total_dom = 0
total_bg = 0
total_misses = 0
count = 0
for d in ['kvasir', 'isic', 'busi', 'pcb']:
    for jf in sorted(glob.glob(os.path.join(out, d, 'combined_f*', 'results.json'))):
        try:
            r = json.load(open(jf))
            total_dom += r['miss_decomposition']['domain_misses']
            total_bg += r['miss_decomposition']['background_misses']
            total_misses += r['miss_decomposition']['total_misses']
            count += 1
        except:
            pass

if total_misses > 0:
    print()
    print('  AGGREGATE (all domains, all MLP fractions):')
    print(f'    Total runs:            {count}')
    print(f'    Total misses:          {total_misses:,}')
    print(f'    Domain misses:         {total_dom:,} ({total_dom/total_misses*100:.2f}%)')
    print(f'    Background misses:     {total_bg:,} ({total_bg/total_misses*100:.2f}%)')
    print()
    print(f'    -> {total_bg/total_misses*100:.1f}% of missed masks are background (irrelevant)')
    print(f'    -> Only {total_dom/total_misses*100:.2f}% are domain-relevant objects')
    print()
print('=' * 135)
print()
PYEOF
