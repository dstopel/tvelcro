"""Structured head removal (true slicing) for timm vit_base_patch16_224.

build_sliced_vit(model, manifest) replaces each affected block's Attention
with a SlicedVitAttention computing ONLY the kept heads: the fused qkv Linear
rows (q/k/v thirds, contiguous per-head d_h slices) and the proj input columns
are sliced, and the frozen heads' proj contribution
C_b = sum_h mean_h @ W_proj[:, slice_h].T is precomputed and added after proj.

Unlike SAM, timm's Attention.forward reshapes its output back to the input
channel count, which breaks under a reduced head count, so SlicedVitAttention
carries its own (otherwise identical) forward.

Also provides ViTFreezeHookManager, the hook-based mean-replacement path
(mirrors sam_velcro_pkg.hooks.FreezeHookManager), used as the "hooked"
benchmark variant and as the self-test reference.

Self-test (CPU, no checkpoint needed):
  python3 vit/surgery.py --selftest
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


class SlicedVitAttention(nn.Module):
    """Head-sliced timm-style attention plus the frozen heads' constant term."""

    def __init__(self, qkv: Optional[nn.Linear], proj: Optional[nn.Linear],
                 num_heads: int, head_dim: int,
                 q_norm: nn.Module, k_norm: nn.Module,
                 const_full: torch.Tensor, const_pooled: torch.Tensor):
        super().__init__()
        self.qkv = qkv
        self.proj = proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.register_buffer("const_full", const_full)
        self.register_buffer("const_pooled", const_pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        if self.qkv is not None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
            out = self.proj(out)
        else:
            out = torch.zeros(B, N, self.const_pooled.shape[-1],
                              dtype=x.dtype, device=x.device)
        if N == self.const_full.shape[0]:
            out = out + self.const_full.to(out.dtype)
        else:
            out = out + self.const_pooled.to(out.dtype)
        return out


def slice_vit_attention(attn: nn.Module,
                        frozen_means: Dict[int, np.ndarray]) -> SlicedVitAttention:
    """Slice one timm Attention module: keep only non-frozen heads."""
    num_heads = attn.num_heads
    dim = attn.proj.out_features
    head_dim = getattr(attn, "head_dim", dim // num_heads)
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
    else:
        qkv = proj = None
        if attn.proj.bias is not None:
            const = const + attn.proj.bias.detach().float().cpu()

    # qk-norm operates per head_dim, independent of head count; copy if present.
    q_norm = copy.deepcopy(getattr(attn, "q_norm", None)) or nn.Identity()
    k_norm = copy.deepcopy(getattr(attn, "k_norm", None)) or nn.Identity()

    return SlicedVitAttention(qkv, proj, len(kept), head_dim,
                              q_norm, k_norm, const, const.mean(dim=0))


def build_sliced_vit(model, manifest):
    """Replace attention modules in a timm ViT per the freeze manifest, in place.

    manifest: dict with "frozen_heads": [{"block": b, "head": h, "mean_path": ...}]
    (same format as the SAM freeze manifest) or a path to that JSON.
    """
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
        blk = model.blocks[b]
        device = blk.attn.proj.weight.device
        dtype = blk.attn.proj.weight.dtype
        blk.attn = slice_vit_attention(blk.attn, means).to(device=device, dtype=dtype)
    return model


class ViTFreezeHookManager:
    """Pre-hooks on blocks[b].attn.proj replacing frozen head slots with means.

    Mirrors sam_velcro_pkg.hooks.FreezeHookManager (including mean pooling on
    sequence-length mismatch) for the timm (B, N, C) tensor layout.
    """

    def __init__(self, model, frozen_heads: Dict[Tuple[int, int], np.ndarray]):
        self.model = model
        self.hooks = []
        self._per_block: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        for (b, h), mean in frozen_heads.items():
            self._per_block[b][h] = torch.from_numpy(np.asarray(mean, dtype=np.float32))

    def register(self) -> None:
        for b, head_means in self._per_block.items():
            attn = self.model.blocks[b].attn
            num_heads = attn.num_heads
            head_dim = attn.proj.in_features // num_heads
            device = attn.proj.weight.device
            means = {h: m.to(device) for h, m in head_means.items()}
            self.hooks.append(attn.proj.register_forward_pre_hook(
                self._make_hook(means, num_heads, head_dim)))

    @staticmethod
    def _make_hook(head_means, num_heads, head_dim):
        def hook(module, args):
            x = args[0]  # (B, N, C)
            B, N, C = x.shape
            reshaped = x.view(B, N, num_heads, head_dim).clone()
            for h, mean_t in head_means.items():
                if mean_t.shape[0] == N:
                    reshaped[:, :, h, :] = mean_t.to(x.dtype).unsqueeze(0)
                else:
                    pooled = mean_t.mean(dim=0, keepdim=True).expand(N, -1)
                    reshaped[:, :, h, :] = pooled.to(x.dtype).unsqueeze(0)
            return (reshaped.view(B, N, C),)
        return hook

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Local timm-style Attention (self-test fallback when timm is absent).
# ---------------------------------------------------------------------------

class _LocalVitAttention(nn.Module):
    """Faithful copy of timm.models.vision_transformer.Attention (eval path)."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Self-test: sliced path vs hook path, numerical equivalence.
# ---------------------------------------------------------------------------

def _selftest(seed: int = 0) -> None:
    try:
        from timm.models.vision_transformer import Attention

        def make_attn():
            return Attention(dim=768, num_heads=12, qkv_bias=True).eval()
        src = "timm.Attention"
    except Exception:
        def make_attn():
            return _LocalVitAttention(dim=768, num_heads=12, qkv_bias=True).eval()
        src = "local reimplementation"

    torch.manual_seed(seed)
    num_heads, head_dim, n_tokens = 12, 64, 197
    cases = [
        ("N-match means", {1: n_tokens, 4: n_tokens, 9: n_tokens}),
        ("pooled means (N mismatch)", {0: 50, 6: 50}),
        ("all heads frozen", {h: n_tokens for h in range(num_heads)}),
    ]

    print(f"ViT surgery self-test (Attention source: {src})")
    for name, spec in cases:
        attn = make_attn()
        x = torch.randn(2, n_tokens, num_heads * head_dim)
        means = {h: torch.randn(s, head_dim).numpy() for h, s in spec.items()}

        # Reference: hook path on a copy of the module.
        ref_attn = copy.deepcopy(attn)
        fake_model = SimpleNamespace(blocks=[SimpleNamespace(attn=ref_attn)])
        mgr = ViTFreezeHookManager(fake_model, {(0, h): m for h, m in means.items()})
        mgr.register()
        with torch.inference_mode():
            out_hook = ref_attn(x)
        mgr.remove()

        # Structured path.
        sliced = slice_vit_attention(copy.deepcopy(attn), means)
        with torch.inference_mode():
            out_sliced = sliced(x)

        diff = (out_hook - out_sliced).abs().max().item()
        assert diff < 1e-4, f"{name}: max abs diff {diff:.3e} >= 1e-4"
        kept = num_heads - len(spec)
        print(f"  PASS {name:<28} kept={kept:>2}/{num_heads}  max abs diff {diff:.3e}")
    print("All ViT surgery self-tests passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ViT structured head-removal surgery")
    p.add_argument("--selftest", action="store_true",
                   help="Run numerical equivalence test vs the hook path")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.selftest:
        _selftest(args.seed)
    else:
        p.print_help()
