#!/bin/bash
# =============================================================================
# exp_profile_size.sh - Profile storage and hook-overhead report
#
# Runs sam/report_profile_size.py over every profile directory found under
# $OUT_ROOT: total on-disk size, frozen-heads-only size, FP16-converted size,
# and a CPU micro-benchmark of freeze hook attach/detach time.
#
# Run from repo root:
#   bash experiments/sam/exp_profile_size.sh
#   # Restrict to one profile dir and manifest:
#   PROFILE_DIRS=results/sam_main_sweep/profile \
#     FREEZE_MANIFEST=results/sam_main_sweep/freeze_kvasir_f0.25/freeze_manifest.json \
#     bash experiments/sam/exp_profile_size.sh
#
# ETA: seconds per profile dir; CPU only, no GPU or checkpoint needed.
# =============================================================================
set -euo pipefail

# --- Config (env-overridable) ------------------------------------------------
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_ROOT="${OUT_ROOT:-./results}"
PROFILE_DIRS="${PROFILE_DIRS:-}"
FREEZE_MANIFEST="${FREEZE_MANIFEST:-}"

export PYTHONPATH=.

if [ ! -f sam/report_profile_size.py ]; then
    echo "ERROR: run from repo root (sam/report_profile_size.py not found)" >&2
    exit 1
fi

# Auto-discover profile dirs: any directory containing <group>/block0_head0_mean.npy
if [ -z "$PROFILE_DIRS" ]; then
    if [ ! -d "$OUT_ROOT" ]; then
        echo "ERROR: no results at $OUT_ROOT and no PROFILE_DIRS given" >&2
        exit 1
    fi
    PROFILE_DIRS="$(find "$OUT_ROOT" -name 'block0_head0_mean.npy' \
        | xargs -n1 dirname 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u)"
fi

if [ -z "$PROFILE_DIRS" ]; then
    echo "ERROR: no profiles found under $OUT_ROOT (run an experiment first)" >&2
    exit 1
fi

for PROF in $PROFILE_DIRS; do
    echo ""
    echo "############ $PROF ############"

    # Pick a manifest for the frozen-only report: explicit env var, else the
    # first freeze_manifest.json in the same results directory.
    MANIFEST="$FREEZE_MANIFEST"
    if [ -z "$MANIFEST" ]; then
        MANIFEST="$(find "$(dirname "$PROF")" -maxdepth 2 -name freeze_manifest.json 2>/dev/null | sort | head -1 || true)"
    fi

    if [ -n "$MANIFEST" ]; then
        python3 sam/report_profile_size.py --profile_dir "$PROF" --freeze_manifest "$MANIFEST"
    else
        echo "(no freeze manifest found near $PROF; reporting sizes only)"
        python3 sam/report_profile_size.py --profile_dir "$PROF"
    fi
done
