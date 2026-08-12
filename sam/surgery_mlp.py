"""Structured removal of frozen MLP neurons in the SAM ViT-H encoder.

A frozen neuron's post-GELU activation is a per-neuron scalar mean, so its
contribution through lin2 is mean_j * W2[:, j]: a constant vector that folds
directly into lin2's bias. Slicing therefore deletes the neuron's row from
lin1 (weight + bias) and its column from W2, and adds the folded constant to
lin2's bias. No runtime additions remain: the sliced MLP is a genuinely
smaller pair of matmuls.

Selection reuses sam_velcro_pkg.mlp_hooks.select_mlp_neurons_to_freeze so the
sliced model freezes exactly the neurons the hook-based accuracy runs froze.

Self-test (CPU): PYTHONPATH=. python sam/surgery_mlp.py --selftest
"""
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_sliced_sam_mlp(sam_model, frozen_neurons: Dict[Tuple[int, int], float]):
    """Slice frozen MLP neurons out of the encoder in place. Returns the model."""
    per_block = defaultdict(dict)
    for (b, j), mean in frozen_neurons.items():
        per_block[b][j] = float(mean)

    for b, means in sorted(per_block.items()):
        mlp = sam_model.image_encoder.blocks[b].mlp
        lin1, lin2 = mlp.lin1, mlp.lin2
        D = lin1.out_features
        kept = [j for j in range(D) if j not in means]
        if not kept:
            raise NotImplementedError(f"block {b}: all {D} MLP neurons frozen")
        dev, dtype = lin2.weight.device, lin2.weight.dtype
        idx = torch.tensor(kept, dtype=torch.long, device=dev)

        # fold frozen contributions into lin2 bias, from the ORIGINAL weight
        w2 = lin2.weight.detach()  # (hidden, D)
        fold = torch.zeros(w2.shape[0], dtype=dtype, device=dev)
        for j, m in means.items():
            fold += m * w2[:, j]

        new1 = nn.Linear(lin1.in_features, len(kept),
                         bias=lin1.bias is not None, device=dev, dtype=dtype)
        new1.weight = nn.Parameter(lin1.weight.detach()[idx].clone())
        if lin1.bias is not None:
            new1.bias = nn.Parameter(lin1.bias.detach()[idx].clone())

        new2 = nn.Linear(len(kept), lin2.out_features,
                         bias=True, device=dev, dtype=dtype)
        new2.weight = nn.Parameter(w2[:, idx].clone())
        base_bias = (lin2.bias.detach().clone() if lin2.bias is not None
                     else torch.zeros(w2.shape[0], dtype=dtype, device=dev))
        new2.bias = nn.Parameter(base_bias + fold)

        mlp.lin1, mlp.lin2 = new1, new2
    return sam_model


# --------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)

    class _MLP(nn.Module):  # mirrors segment_anything MLPBlock
        def __init__(self, dim=32, hidden=128):
            super().__init__()
            self.lin1 = nn.Linear(dim, hidden)
            self.lin2 = nn.Linear(hidden, dim)
            self.act = nn.GELU()

        def forward(self, x):
            return self.lin2(self.act(self.lin1(x)))

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP()

        def forward(self, x):
            return x + self.mlp(x)

    class _SAM(nn.Module):
        def __init__(self, n=3):
            super().__init__()
            enc = nn.Module()
            enc.blocks = nn.ModuleList(_Block() for _ in range(n))
            self.image_encoder = enc

        def forward(self, x):
            for b in self.image_encoder.blocks:
                x = b(x)
            return x

    rng = np.random.default_rng(0)
    cases = [
        ("few neurons one block", {(0, 5): 0.7, (0, 100): -0.2}),
        ("many neurons across blocks",
         {(b, j): float(rng.normal()) for b in range(3)
          for j in rng.choice(128, size=48, replace=False)}),
    ]
    for name, frozen in cases:
        model = _SAM().eval()
        x = torch.randn(2, 14, 14, 32)  # (B, H, W, C) like SAM windowed blocks

        # reference: hook-based mean replacement at lin2 input (post-GELU)
        handles = []
        per_block = defaultdict(dict)
        for (b, j), m in frozen.items():
            per_block[b][j] = m
        for b, means in per_block.items():
            lin2 = model.image_encoder.blocks[b].mlp.lin2

            def hook(_mod, args, _means=means):
                h = args[0].clone()
                for j, m in _means.items():
                    h[..., j] = m
                return (h,)
            handles.append(lin2.register_forward_pre_hook(hook))
        with torch.no_grad():
            ref = model(x).clone()
        for h in handles:
            h.remove()

        build_sliced_sam_mlp(model, frozen)
        with torch.no_grad():
            sliced = model(x)
        diff = (ref - sliced).abs().max().item()
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"  {status} {name:<30} max abs diff {diff:.3e}")
        assert diff < 1e-4, name
    print("All SAM MLP surgery self-tests passed.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
