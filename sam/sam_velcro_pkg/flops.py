"""Accurate FLOP calculation for SAM ViT-H attention head freezing.

Computes per-head savings accounting for windowed vs global attention blocks.
Windowed blocks process 25 windows of 196 tokens each (14x14).
Global blocks process 4096 tokens (64x64).

Per attention block, the operations are:
  QKV projection:  2 * S_total * C * 3C       (linear, scales with token count)
  Q @ K^T:         2 * S^2 * C * N_windows    (quadratic, scales with window size)
  Attn @ V:        2 * S^2 * C * N_windows    (quadratic, same)
  Output proj:     2 * S_total * C * C         (linear)
  MLP:             4 * S_total * C * MLP_DIM   (not affected by head freezing)

Freezing a head (1/H of attention) saves 1/H of QKV, Q@K^T, Attn@V, and output proj.
MLP is never saved — it operates on the full hidden dimension regardless.
"""

from typing import Dict, List

from .constants import (
    GLOBAL_ATTN_INDEXES,
    HEAD_DIM,
    HIDDEN_DIM,
    MLP_DIM,
    NUM_BLOCKS,
    NUM_HEADS,
    NUM_WINDOWS,
    TOKENS_GLOBAL,
    TOKENS_WINDOW,
)

C = HIDDEN_DIM      # 1280
H = NUM_HEADS        # 16
D_h = HEAD_DIM       # 80


def _attn_flops_windowed() -> int:
    """Total attention FLOPs for one windowed block."""
    S = TOKENS_WINDOW                              # 196
    N = NUM_WINDOWS                                 # 25
    S_total = S * N                                 # 4900 (padded 70x70)
    qkv = 2 * S_total * C * 3 * C                  # QKV projection
    qk = N * 2 * H * S * D_h * S                   # Q@K^T per window
    av = N * 2 * H * S * D_h * S                   # Attn@V per window
    out = 2 * S_total * C * C                       # Output projection
    return qkv + qk + av + out


def _attn_flops_global() -> int:
    """Total attention FLOPs for one global block."""
    S = TOKENS_GLOBAL                               # 4096
    qkv = 2 * S * C * 3 * C
    qk = 2 * H * S * D_h * S
    av = 2 * H * S * D_h * S
    out = 2 * S * C * C
    return qkv + qk + av + out


def _mlp_flops() -> int:
    """MLP FLOPs per block (always 4096 tokens, unaffected by windowing)."""
    S = TOKENS_GLOBAL                               # 4096
    return 2 * S * C * MLP_DIM + 2 * S * MLP_DIM * C


def _per_head_savings_windowed() -> int:
    """FLOPs saved by freezing one head in a windowed block."""
    S = TOKENS_WINDOW
    N = NUM_WINDOWS
    S_total = S * N
    qkv = 2 * S_total * D_h * 3 * C                # 1/H of QKV (project to one head's dims)
    qk = N * 2 * S * D_h * S                        # Q@K^T for one head
    av = N * 2 * S * D_h * S                         # Attn@V for one head
    out = 2 * S_total * D_h * C                      # 1/H of output projection
    return qkv + qk + av + out


def _per_head_savings_global() -> int:
    """FLOPs saved by freezing one head in a global block."""
    S = TOKENS_GLOBAL
    qkv = 2 * S * D_h * 3 * C
    qk = 2 * S * D_h * S
    av = 2 * S * D_h * S
    out = 2 * S * D_h * C
    return qkv + qk + av + out


ATTN_FLOPS_WINDOWED = _attn_flops_windowed()
ATTN_FLOPS_GLOBAL = _attn_flops_global()
MLP_FLOPS = _mlp_flops()
PER_HEAD_SAVINGS_WINDOWED = _per_head_savings_windowed()
PER_HEAD_SAVINGS_GLOBAL = _per_head_savings_global()

N_WINDOWED_BLOCKS = NUM_BLOCKS - len(GLOBAL_ATTN_INDEXES)
N_GLOBAL_BLOCKS = len(GLOBAL_ATTN_INDEXES)

TOTAL_ATTN_FLOPS = (N_WINDOWED_BLOCKS * ATTN_FLOPS_WINDOWED +
                    N_GLOBAL_BLOCKS * ATTN_FLOPS_GLOBAL)
TOTAL_MLP_FLOPS = NUM_BLOCKS * MLP_FLOPS
TOTAL_ENCODER_FLOPS = TOTAL_ATTN_FLOPS + TOTAL_MLP_FLOPS


def compute_flop_reduction(frozen_heads: List[Dict]) -> Dict:
    """Compute accurate FLOP reduction from a list of frozen head entries.

    Each entry must have 'block' (int) and 'block_type' ('windowed'|'global').

    Returns dict with:
        saved_flops: absolute FLOPs saved
        attention_flop_reduction_pct: % of attention FLOPs saved
        encoder_flop_reduction_pct: % of total encoder FLOPs saved
        windowed_heads_frozen: count
        global_heads_frozen: count
        per_head_windowed_gflops: cost of one windowed head
        per_head_global_gflops: cost of one global head
    """
    n_windowed = 0
    n_global = 0
    for entry in frozen_heads:
        block_idx = entry["block"]
        if block_idx in GLOBAL_ATTN_INDEXES:
            n_global += 1
        else:
            n_windowed += 1

    saved = (n_windowed * PER_HEAD_SAVINGS_WINDOWED +
             n_global * PER_HEAD_SAVINGS_GLOBAL)

    return {
        "saved_flops": saved,
        "attention_flop_reduction_pct": saved / TOTAL_ATTN_FLOPS * 100 if TOTAL_ATTN_FLOPS > 0 else 0,
        "encoder_flop_reduction_pct": saved / TOTAL_ENCODER_FLOPS * 100 if TOTAL_ENCODER_FLOPS > 0 else 0,
        "windowed_heads_frozen": n_windowed,
        "global_heads_frozen": n_global,
        "total_heads_frozen": n_windowed + n_global,
        "saved_gflops": saved / 1e9,
        "total_attn_gflops": TOTAL_ATTN_FLOPS / 1e9,
        "total_encoder_gflops": TOTAL_ENCODER_FLOPS / 1e9,
        "per_head_windowed_gflops": PER_HEAD_SAVINGS_WINDOWED / 1e9,
        "per_head_global_gflops": PER_HEAD_SAVINGS_GLOBAL / 1e9,
    }


def compute_flop_reduction_from_manifest(manifest: Dict) -> Dict:
    """Compute FLOP reduction from a freeze manifest JSON."""
    return compute_flop_reduction(manifest.get("frozen_heads", []))


def _per_neuron_savings_windowed() -> int:
    """FLOPs saved by freezing one MLP neuron in a windowed block."""
    S_total = TOKENS_WINDOW * NUM_WINDOWS  # 196 * 25 = 4900
    lin1_save = 2 * S_total * HIDDEN_DIM   # skip one column of lin1
    lin2_save = 2 * S_total * HIDDEN_DIM   # skip one row of lin2
    return lin1_save + lin2_save


def _per_neuron_savings_global() -> int:
    """FLOPs saved by freezing one MLP neuron in a global block."""
    S = TOKENS_GLOBAL  # 4096
    lin1_save = 2 * S * HIDDEN_DIM
    lin2_save = 2 * S * HIDDEN_DIM
    return lin1_save + lin2_save


PER_NEURON_SAVINGS_WINDOWED = _per_neuron_savings_windowed()
PER_NEURON_SAVINGS_GLOBAL = _per_neuron_savings_global()


def compute_mlp_flop_reduction(frozen_neurons_per_block: Dict[int, int]) -> Dict:
    """Compute MLP FLOP reduction from per-block frozen neuron counts.

    Args:
        frozen_neurons_per_block: dict of {block_idx: num_frozen_neurons}

    Returns dict with saved_flops, mlp/encoder reduction %, etc.
    """
    n_windowed_neurons = 0
    n_global_neurons = 0
    for block_idx, count in frozen_neurons_per_block.items():
        if block_idx in GLOBAL_ATTN_INDEXES:
            n_global_neurons += count
        else:
            n_windowed_neurons += count

    saved = (n_windowed_neurons * PER_NEURON_SAVINGS_WINDOWED +
             n_global_neurons * PER_NEURON_SAVINGS_GLOBAL)

    return {
        "saved_flops": saved,
        "mlp_flop_reduction_pct": saved / TOTAL_MLP_FLOPS * 100 if TOTAL_MLP_FLOPS > 0 else 0,
        "encoder_flop_reduction_pct": saved / TOTAL_ENCODER_FLOPS * 100 if TOTAL_ENCODER_FLOPS > 0 else 0,
        "windowed_neurons_frozen": n_windowed_neurons,
        "global_neurons_frozen": n_global_neurons,
        "total_neurons_frozen": n_windowed_neurons + n_global_neurons,
        "saved_gflops": saved / 1e9,
    }


def compute_combined_flop_reduction(
    attn_manifest: Dict,
    mlp_frozen_neurons: Dict[tuple, float],
) -> Dict:
    """Compute combined attention + MLP FLOP reduction.

    Args:
        attn_manifest: freeze manifest dict (with "frozen_heads" list)
        mlp_frozen_neurons: dict of (block_idx, neuron_idx) -> mean_val

    Returns combined FLOP savings.
    """
    attn_result = compute_flop_reduction_from_manifest(attn_manifest)

    # Count MLP neurons per block
    neurons_per_block: Dict[int, int] = {}
    for (block_idx, _neuron_idx) in mlp_frozen_neurons.keys():
        neurons_per_block[block_idx] = neurons_per_block.get(block_idx, 0) + 1

    mlp_result = compute_mlp_flop_reduction(neurons_per_block)

    total_saved = attn_result["saved_flops"] + mlp_result["saved_flops"]

    return {
        "attn_saved_gflops": attn_result["saved_gflops"],
        "attn_encoder_pct": attn_result["encoder_flop_reduction_pct"],
        "mlp_saved_gflops": mlp_result["saved_gflops"],
        "mlp_encoder_pct": mlp_result["encoder_flop_reduction_pct"],
        "total_saved_gflops": total_saved / 1e9,
        "total_encoder_pct": total_saved / TOTAL_ENCODER_FLOPS * 100,
        "total_encoder_gflops": TOTAL_ENCODER_FLOPS / 1e9,
    }

