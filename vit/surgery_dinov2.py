"""Structured removal for DINOv2 (HF transformers) attention heads.

DINOv2 uses separate query/key/value Linear layers per block (not the fused
qkv of timm ViTs), so slicing differs from vit/surgery.py:
  - remove each frozen head's rows from query/key/value weights and biases,
  - reduce num_attention_heads / all_head_size on the self-attention module,
  - remove the matching input columns of attention.output.dense,
  - add the frozen heads' constant contribution C = sum_h mean_h @ W_dense[:, h].T
    after output.dense (position-dependent, precomputed once).

Also provides Dinov2FreezeHookManager, the hook-based mean-replacement variant
used as the accuracy-equivalence reference and the "hooked" benchmark arm.

Self-test (CPU, no checkpoint needed):
  PYTHONPATH=. python vit/surgery_dinov2.py --selftest
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def _iter_layers(model):
    """Yield DINOv2 encoder layers for either bare or classification models."""
    core = getattr(model, "dinov2", model)
    return core.encoder.layer


class Dinov2FreezeHookManager:
    """Pre-hook on attention.output.dense: overwrite frozen head slices with means."""

    def __init__(self, model, frozen_heads: Dict[tuple, np.ndarray]):
        self.model = model
        self.per_block = defaultdict(dict)
        for (b, h), mean in frozen_heads.items():
            self.per_block[b][h] = mean
        self.handles = []

    def register(self):
        layers = _iter_layers(self.model)
        for b, heads in self.per_block.items():
            attn = layers[b].attention.attention
            dh = attn.attention_head_size
            dense = layers[b].attention.output.dense
            means = {h: torch.as_tensor(m, dtype=dense.weight.dtype,
                                        device=dense.weight.device)
                     for h, m in heads.items()}

            def hook(_mod, inputs, _means=means, _dh=dh):
                x = inputs[0].clone()
                for h, m in _means.items():
                    mm = m
                    if mm.shape[0] != x.shape[1]:  # sequence-length mismatch
                        mm = mm.mean(dim=0, keepdim=True).expand(x.shape[1], -1)
                    x[:, :, h * _dh:(h + 1) * _dh] = mm
                return (x,)

            self.handles.append(dense.register_forward_pre_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class _DenseWithConst(nn.Module):
    """output.dense wrapper adding the frozen heads' precomputed contribution."""

    def __init__(self, dense: nn.Linear, const: torch.Tensor):
        super().__init__()
        self.dense = dense
        self.register_buffer("const", const)

    def forward(self, x):
        y = self.dense(x)
        c = self.const
        if y.shape[1] != c.shape[0]:
            c = c.mean(dim=0, keepdim=True)
        return y + c


def build_sliced_dinov2(model, manifest):
    """Slice frozen heads out of a DINOv2 model in place. Returns the model."""
    manifest_dir = None
    if not isinstance(manifest, dict):
        manifest_dir = Path(manifest).resolve().parent
        with open(manifest) as f:
            manifest = json.load(f)

    def _resolve(stored):
        p = Path(manifest.get("profile_root", ".")) / stored
        if not p.exists() and manifest_dir is not None and (manifest_dir / stored).exists():
            return manifest_dir / stored
        return p

    per_block = defaultdict(dict)
    for e in manifest["frozen_heads"]:
        per_block[e["block"]][e["head"]] = np.load(_resolve(e["mean_path"]))

    layers = _iter_layers(model)
    for b, means in sorted(per_block.items()):
        layer = layers[b]
        attn = layer.attention.attention
        out = layer.attention.output
        H, dh = attn.num_attention_heads, attn.attention_head_size
        frozen = sorted(means)
        kept = [h for h in range(H) if h not in means]
        if not kept:
            raise NotImplementedError(
                f"block {b}: all {H} heads frozen; constant-only layers not supported")

        dev = out.dense.weight.device
        dtype = out.dense.weight.dtype

        # constant contribution of frozen heads, from the ORIGINAL dense weight
        const = torch.zeros(0)
        w_dense = out.dense.weight.detach()  # (hidden, all_head_size)
        for h in frozen:
            m = torch.as_tensor(means[h], dtype=dtype, device=dev)
            contrib = m @ w_dense[:, h * dh:(h + 1) * dh].T  # (S, hidden)
            const = contrib if const.numel() == 0 else const + contrib

        idx = torch.tensor([h * dh + j for h in kept for j in range(dh)],
                           dtype=torch.long, device=dev)
        for lin in (attn.query, attn.key, attn.value):
            lin.weight = nn.Parameter(lin.weight.detach()[idx].clone())
            if lin.bias is not None:
                lin.bias = nn.Parameter(lin.bias.detach()[idx].clone())
            lin.out_features = len(idx)
        attn.num_attention_heads = len(kept)
        attn.all_head_size = len(kept) * dh

        new_dense = nn.Linear(len(idx), out.dense.out_features,
                              bias=out.dense.bias is not None,
                              device=dev, dtype=dtype)
        new_dense.weight = nn.Parameter(w_dense[:, idx].clone())
        if out.dense.bias is not None:
            new_dense.bias = nn.Parameter(out.dense.bias.detach().clone())
        out.dense = _DenseWithConst(new_dense, const)
    return model


# --------------------------------------------------------------------------
def _selftest():
    from transformers import Dinov2Config, Dinov2ForImageClassification
    torch.manual_seed(0)
    cfg = Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       intermediate_size=128, image_size=28, patch_size=14,
                       num_labels=10, attn_implementation="eager")
    dh = cfg.hidden_size // cfg.num_attention_heads
    seq = (cfg.image_size // cfg.patch_size) ** 2 + 1  # patches + CLS

    cases = [
        ("two heads one block", {(0, 1), (0, 3)}),
        ("heads in both blocks", {(0, 0), (1, 2), (1, 3)}),
        ("mean with wrong seq (pooled)", {(1, 1)}),
    ]
    for name, frozen in cases:
        model = Dinov2ForImageClassification(cfg).eval()
        rng = np.random.default_rng(0)
        means = {}
        for (b, h) in frozen:
            s = seq if "pooled" not in name else seq + 3
            means[(b, h)] = rng.normal(size=(s, dh)).astype(np.float32)

        x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
        mgr = Dinov2FreezeHookManager(model, means)
        mgr.register()
        with torch.no_grad():
            ref = model(x).logits.clone()
        mgr.remove()

        manifest = {"profile_root": ".", "frozen_heads": []}
        import tempfile
        tmp = tempfile.mkdtemp()
        for (b, h), m in means.items():
            p = Path(tmp) / f"mean_b{b}h{h}.npy"
            np.save(p, m)
            manifest["frozen_heads"].append(
                {"block": b, "head": h, "mean_path": str(p)})
        build_sliced_dinov2(model, manifest)
        with torch.no_grad():
            sliced = model(x).logits
        diff = (ref - sliced).abs().max().item()
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"  {status} {name:<32} max abs diff {diff:.3e}")
        assert diff < 1e-4, name
    print("All DINOv2 surgery self-tests passed.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
