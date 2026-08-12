"""Structured head removal (true slicing) for SAM ViT-H attention blocks.

build_sliced_sam(sam_model, freeze_manifest) replaces each affected block's
Attention with a SlicedAttention that computes ONLY the kept heads:
  - qkv Linear rows (q/k/v thirds, contiguous per-head d_h slices) sliced to kept heads
  - proj Linear input columns sliced correspondingly, num_heads reduced
  - the frozen heads' proj contribution, a position-dependent constant per block
    C_b = sum_h mean_h @ W_proj[:, slice_h].T, is precomputed and added after proj

SAM's decomposed relative position embeddings (rel_pos_h/rel_pos_w) are shared
across heads, so head slicing leaves them untouched; the original Attention
forward is reused with the reduced head count.

Self-test (CPU, no checkpoint needed):
  python3 sam/surgery.py --selftest
"""

import argparse
import copy
import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlicedAttention(nn.Module):
    """Head-sliced SAM attention plus the frozen heads' constant proj term.

    inner: the original Attention module with sliced qkv/proj and reduced
    num_heads (None when every head in the block is frozen; then the proj
    bias is folded into the constants).
    const_full: (S_mean, dim) used when runtime S == S_mean.
    const_pooled: (dim,) token-average of const_full, used on S mismatch,
    matching the mean pooling in FreezeHookManager exactly.
    """

    def __init__(self, inner: Optional[nn.Module],
                 const_full: torch.Tensor, const_pooled: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("const_full", const_full)
        self.register_buffer("const_pooled", const_pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        if self.inner is not None:
            out = self.inner(x)
        else:
            out = torch.zeros(B, H, W, self.const_pooled.shape[-1],
                              dtype=x.dtype, device=x.device)
        if H * W == self.const_full.shape[0]:
            out = out + self.const_full.to(out.dtype).view(1, H, W, -1)
        else:
            out = out + self.const_pooled.to(out.dtype)
        return out


def slice_sam_attention(attn: nn.Module,
                        frozen_means: Dict[int, np.ndarray]) -> SlicedAttention:
    """Slice one SAM Attention module: keep only non-frozen heads."""
    num_heads = attn.num_heads
    dim = attn.proj.out_features
    head_dim = dim // num_heads
    kept = [h for h in range(num_heads) if h not in frozen_means]

    shapes = {np.asarray(m).shape for m in frozen_means.values()}
    assert len(shapes) == 1, f"mixed mean shapes within one block: {shapes}"
    s_mean = next(iter(shapes))[0]

    w_proj = attn.proj.weight.detach().float().cpu()  # (dim, dim), out x in
    const = torch.zeros(s_mean, dim)
    for h, mean in frozen_means.items():
        mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float32))
        const = const + mean_t @ w_proj[:, h * head_dim:(h + 1) * head_dim].t()

    if kept:
        inner = copy.deepcopy(attn)
        idx = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in kept])
        rows = torch.cat([idx, idx + dim, idx + 2 * dim])
        has_qkv_bias = attn.qkv.bias is not None
        qkv = nn.Linear(attn.qkv.in_features, 3 * len(kept) * head_dim, bias=has_qkv_bias)
        has_proj_bias = attn.proj.bias is not None
        proj = nn.Linear(len(kept) * head_dim, dim, bias=has_proj_bias)
        with torch.no_grad():
            qkv.weight.copy_(attn.qkv.weight[rows, :])
            if has_qkv_bias:
                qkv.bias.copy_(attn.qkv.bias[rows])
            proj.weight.copy_(attn.proj.weight[:, idx])
            if has_proj_bias:
                proj.bias.copy_(attn.proj.bias)
        inner.qkv = qkv
        inner.proj = proj
        inner.num_heads = len(kept)
    else:
        inner = None
        if attn.proj.bias is not None:
            const = const + attn.proj.bias.detach().float().cpu()

    return SlicedAttention(inner, const, const.mean(dim=0))


def build_sliced_sam(sam_model, freeze_manifest):
    """Replace attention modules in sam_model per the freeze manifest, in place.

    freeze_manifest: manifest dict (see sam_velcro_pkg.freeze.build_manifest)
    or a path to freeze_manifest.json. Returns the modified model.
    """
    manifest = freeze_manifest
    manifest_dir = None
    if not isinstance(manifest, dict):
        from pathlib import Path as _P
        manifest_dir = _P(manifest).resolve().parent
        with open(manifest) as f:
            manifest = json.load(f)

    def _resolve(stored):
        from pathlib import Path as _P
        p = _P(manifest.get("profile_root", ".")) / stored
        if not p.exists() and manifest_dir is not None and (manifest_dir / stored).exists():
            return manifest_dir / stored
        return p

    per_block: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for e in manifest["frozen_heads"]:
        per_block[e["block"]][e["head"]] = np.load(_resolve(e["mean_path"]))

    for b, means in sorted(per_block.items()):
        blk = sam_model.image_encoder.blocks[b]
        device = blk.attn.proj.weight.device
        dtype = blk.attn.proj.weight.dtype
        blk.attn = slice_sam_attention(blk.attn, means).to(device=device, dtype=dtype)
    return sam_model


# ---------------------------------------------------------------------------
# Faithful local copy of segment_anything's Attention (used by the self-test
# only when segment_anything is not importable).
# ---------------------------------------------------------------------------

def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def _add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    r_h = _get_rel_pos(q_h, k_h, rel_pos_h)
    r_w = _get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, r_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, r_w)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


class _LocalAttention(nn.Module):
    """Line-for-line port of segment_anything.modeling.image_encoder.Attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 use_rel_pos: bool = False,
                 input_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = _add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Self-test: sliced path vs FreezeHookManager path, numerical equivalence.
# ---------------------------------------------------------------------------

def _selftest(seed: int = 0) -> None:
    from sam_velcro_pkg.constants import HEAD_DIM, HIDDEN_DIM, NUM_HEADS
    from sam_velcro_pkg.hooks import FreezeHookManager

    try:
        from segment_anything.modeling.image_encoder import Attention
        src = "segment_anything.Attention"
    except Exception:
        Attention = _LocalAttention
        src = "local reimplementation"

    torch.manual_seed(seed)
    device = torch.device("cpu")
    # Real head geometry (16 heads x 80) so FreezeHookManager's hardcoded
    # constants apply; small 7x7 spatial grid keeps the CPU test fast.
    grid = 7
    s = grid * grid
    cases = [
        ("S-match means", {2: s, 5: s, 11: s}),
        ("pooled means (S mismatch)", {0: 196, 7: 196}),
        ("all heads frozen", {h: s for h in range(NUM_HEADS)}),
    ]

    print(f"SAM surgery self-test (Attention source: {src})")
    for name, spec in cases:
        attn = Attention(dim=HIDDEN_DIM, num_heads=NUM_HEADS, qkv_bias=True,
                         use_rel_pos=True, input_size=(grid, grid)).eval()
        with torch.no_grad():
            attn.rel_pos_h.normal_(0, 0.02)
            attn.rel_pos_w.normal_(0, 0.02)
        x = torch.randn(2, grid, grid, HIDDEN_DIM)
        means = {h: torch.randn(s_mean, HEAD_DIM).numpy() for h, s_mean in spec.items()}

        # Reference: existing hook path on a copy of the module.
        ref_attn = copy.deepcopy(attn)
        fake_model = SimpleNamespace(image_encoder=SimpleNamespace(
            blocks=[SimpleNamespace(attn=ref_attn)]))
        mgr = FreezeHookManager(fake_model, {(0, h): m for h, m in means.items()}, device)
        mgr.register()
        with torch.inference_mode():
            out_hook = ref_attn(x)
        mgr.remove()

        # Structured path.
        sliced = slice_sam_attention(copy.deepcopy(attn), means)
        with torch.inference_mode():
            out_sliced = sliced(x)

        diff = (out_hook - out_sliced).abs().max().item()
        assert diff < 1e-4, f"{name}: max abs diff {diff:.3e} >= 1e-4"
        kept = NUM_HEADS - len(spec)
        print(f"  PASS {name:<28} kept={kept:>2}/{NUM_HEADS}  max abs diff {diff:.3e}")
    print("All SAM surgery self-tests passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAM structured head-removal surgery")
    p.add_argument("--selftest", action="store_true",
                   help="Run numerical equivalence test vs the hook path")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.selftest:
        _selftest(args.seed)
    else:
        p.print_help()
