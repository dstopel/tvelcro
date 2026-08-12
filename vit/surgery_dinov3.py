"""Structured removal for DINOv3 ViT (HF transformers) attention heads.

DINOv3 (transformers 5.x) differs from DINOv2 in module layout:
  - encoder layers live at <core>.model.layer[b] (core = model.dinov3 for the
    classification wrapper, or the bare DINOv3ViTModel),
  - attention uses separate q_proj/k_proj/v_proj and o_proj Linear layers,
    with head geometry in attn.num_heads / attn.head_dim,
  - RoPE cos/sin tables are computed once per forward at model level with
    shape (num_patches, head_dim) and broadcast over heads, so removing whole
    heads leaves the kept heads' rotary tables untouched (verified numerically
    by --selftest: sliced logits match the hook-based reference).

Slicing per block:
  - remove each frozen head's rows from q/k/v weights (and biases if present),
  - reduce attn.num_heads,
  - remove the matching input columns of attn.o_proj,
  - add the frozen heads' constant contribution C = sum_h mean_h @ W_o[:, h].T
    after o_proj (position-dependent, precomputed once).

Also provides Dinov3FreezeHookManager (hook-based mean-replacement reference)
and DINOv3ViTForImageClassification, the linear-classifier head used by the
dimidagd/dinov3-vit7b16-pretrain-lvd1689m-imagenet1k-lc checkpoint, which is
absent from transformers 5.15 (logits = Linear on [CLS ; mean(patch tokens)]).

Self-test (CPU fp32, no checkpoint needed; tolerance 1e-4 — the fp16 GPU
budget is 1e-3):
  PYTHONPATH=. python vit/surgery_dinov3.py --selftest
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import DINOv3ViTModel
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTPreTrainedModel

from surgery_dinov2 import _DenseWithConst

DINOV3_7B_LC_ID = "dimidagd/dinov3-vit7b16-pretrain-lvd1689m-imagenet1k-lc"


class DINOv3ViTForImageClassification(DINOv3ViTPreTrainedModel):
    """DINOv3 backbone + ImageNet linear classifier (missing from transformers 5.15).

    Matches the LC checkpoint layout (dinov3.* + classifier.*; the built-in
    dinov3_vit legacy renaming maps dinov3.layer.N -> dinov3.model.layer.N).
    Head input is [CLS ; mean(patch tokens)] (2*hidden), registers excluded,
    on the final-norm'd hidden states — the DINOv3 linear-probe protocol.
    """

    base_model_prefix = "dinov3"

    def __init__(self, config):
        super().__init__(config)
        self.dinov3 = DINOv3ViTModel(config)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, pixel_values, **kwargs):
        out = self.dinov3(pixel_values=pixel_values, **kwargs)
        seq = out.last_hidden_state  # (B, 1 + registers + patches, C), post-norm
        n_prefix = 1 + self.config.num_register_tokens
        pooled = torch.cat([seq[:, 0], seq[:, n_prefix:].mean(dim=1)], dim=-1)
        return ImageClassifierOutput(logits=self.classifier(pooled))


def _iter_layers(model):
    """Yield DINOv3 encoder layers for either bare or classification models."""
    core = getattr(model, "dinov3", model)
    return core.model.layer


class Dinov3FreezeHookManager:
    """Pre-hook on attention.o_proj: overwrite frozen head slices with means."""

    def __init__(self, model, frozen_heads: Dict[tuple, np.ndarray]):
        self.model = model
        self.per_block = defaultdict(dict)
        for (b, h), mean in frozen_heads.items():
            self.per_block[b][h] = mean
        self.handles = []

    def register(self):
        layers = _iter_layers(self.model)
        for b, heads in self.per_block.items():
            attn = layers[b].attention
            dh = attn.head_dim
            o_proj = attn.o_proj
            means = {h: torch.as_tensor(m, dtype=o_proj.weight.dtype,
                                        device=o_proj.weight.device)
                     for h, m in heads.items()}

            def hook(_mod, inputs, _means=means, _dh=dh):
                x = inputs[0].clone()
                for h, m in _means.items():
                    mm = m
                    if mm.shape[0] != x.shape[1]:  # sequence-length mismatch
                        mm = mm.mean(dim=0, keepdim=True).expand(x.shape[1], -1)
                    x[:, :, h * _dh:(h + 1) * _dh] = mm
                return (x,)

            self.handles.append(o_proj.register_forward_pre_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_sliced_dinov3(model, manifest):
    """Slice frozen heads out of a DINOv3 model in place. Returns the model."""
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
        attn = layers[b].attention
        H, dh = attn.num_heads, attn.head_dim
        frozen = sorted(means)
        kept = [h for h in range(H) if h not in means]
        if not kept:
            raise NotImplementedError(
                f"block {b}: all {H} heads frozen; constant-only layers not supported")

        dev = attn.o_proj.weight.device
        dtype = attn.o_proj.weight.dtype

        # constant contribution of frozen heads, from the ORIGINAL o_proj weight
        const = torch.zeros(0)
        w_o = attn.o_proj.weight.detach()  # (hidden, H*dh)
        for h in frozen:
            m = torch.as_tensor(means[h], dtype=dtype, device=dev)
            contrib = m @ w_o[:, h * dh:(h + 1) * dh].T  # (S, hidden)
            const = contrib if const.numel() == 0 else const + contrib

        idx = torch.tensor([h * dh + j for h in kept for j in range(dh)],
                           dtype=torch.long, device=dev)
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj):
            lin.weight = nn.Parameter(lin.weight.detach()[idx].clone())
            if lin.bias is not None:
                lin.bias = nn.Parameter(lin.bias.detach()[idx].clone())
            lin.out_features = len(idx)
        attn.num_heads = len(kept)

        new_o = nn.Linear(len(idx), attn.o_proj.out_features,
                          bias=attn.o_proj.bias is not None,
                          device=dev, dtype=dtype)
        new_o.weight = nn.Parameter(w_o[:, idx].clone())
        if attn.o_proj.bias is not None:
            new_o.bias = nn.Parameter(attn.o_proj.bias.detach().clone())
        attn.o_proj = _DenseWithConst(new_o, const)
    return model


# --------------------------------------------------------------------------
def _selftest():
    from transformers import DINOv3ViTConfig
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=128, image_size=32, patch_size=16,
                          num_register_tokens=4, num_labels=10,
                          attn_implementation="eager")
    dh = cfg.hidden_size // cfg.num_attention_heads
    seq = (cfg.image_size // cfg.patch_size) ** 2 + 1 + cfg.num_register_tokens

    cases = [
        ("two heads one block", {(0, 1), (0, 3)}),
        ("heads in both blocks", {(0, 0), (1, 2), (1, 3)}),
        ("mean with wrong seq (pooled)", {(1, 1)}),
    ]
    for name, frozen in cases:
        model = DINOv3ViTForImageClassification(cfg).eval()
        rng = np.random.default_rng(0)
        means = {}
        for (b, h) in frozen:
            s = seq if "pooled" not in name else seq + 3
            means[(b, h)] = rng.normal(size=(s, dh)).astype(np.float32)

        x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
        mgr = Dinov3FreezeHookManager(model, means)
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
        build_sliced_dinov3(model, manifest)
        with torch.no_grad():
            sliced = model(x).logits
        diff = (ref - sliced).abs().max().item()
        # CPU fp32 bar; the fp16 GPU equivalence budget is 1e-3
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"  {status} {name:<32} max abs diff {diff:.3e}")
        assert diff < 1e-4, name
    print("All DINOv3 surgery self-tests passed.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
