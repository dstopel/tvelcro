#!/bin/bash
# analyze_freeze_params.sh — Analyze profile data to recommend freeze parameters per domain
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
PROFILE_DIR="${PROFILE_DIR:-./results_cross_domain/profile}"
EXTRA_PROFILE_DIR="${EXTRA_PROFILE_DIR:-./results_wave2/profile}"
IOU_TARGET="${IOU_TARGET:-0.975}"
OUT_JSON="${OUT_JSON:-./results_freeze_recommendations.json}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -d "$PROFILE_DIR" ]; then
    echo "ERROR: Primary profile directory not found: $PROFILE_DIR"
    echo "  Set PROFILE_DIR to the directory containing general/ and domain subdirectories."
    exit 1
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD="python sam_velcro.py"
# Detect python binary
if ! command -v python &>/dev/null; then
    CMD="python3 sam_velcro.py"
fi

ARGS=(
    "scripts/analyze_freeze_params.py"
    --profile_dir "$PROFILE_DIR"
    --iou_target "$IOU_TARGET"
    --out_json "$OUT_JSON"
)

if [ -d "$EXTRA_PROFILE_DIR" ]; then
    ARGS+=(--extra_profile_dir "$EXTRA_PROFILE_DIR")
fi

# Pass through any extra arguments
ARGS+=("$@")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running freeze parameter analysis..."
echo "  Profile dir:       $PROFILE_DIR"
if [ -d "$EXTRA_PROFILE_DIR" ]; then
    echo "  Extra profile dir: $EXTRA_PROFILE_DIR"
fi
echo "  IoU target:        $IOU_TARGET"
echo "  Output:            $OUT_JSON"
echo ""

${CMD%sam_velcro.py} "${ARGS[@]}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Recommendations saved to: $OUT_JSON"
