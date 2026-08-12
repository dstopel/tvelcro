"""Latency and energy benchmark: baseline vs hooked vs sliced timm ViT-Base.

Measures full-model forward latency (ms) and throughput (images/s) at batch
sizes given by --batch_sizes, plus energy per forward when NVML is available.

Run from repo root with PYTHONPATH=. (see experiments/vit/exp_latency.sh):
  PYTHONPATH=. python3 vit/bench_latency.py --freeze_manifest ...

Note: the sliced variant mutates the loaded model in place, so it always runs last.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.timing import measure_energy, measure_latency
from surgery import ViTFreezeHookManager, build_sliced_vit


def load_frozen_heads(manifest):
    from pathlib import Path
    root = Path(manifest.get("profile_root", "."))
    return {(e["block"], e["head"]): np.load(root / e["mean_path"])
            for e in manifest["frozen_heads"]}


def bench_variant(name, model, batch_sizes, args, device, input_size=224):
    result = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, input_size, input_size, device=device)

        def fwd():
            model(x)

        print(f"[{name}] batch {bs} ...")
        lat = measure_latency(fwd, warmup=args.warmup, iters=args.iters,
                              device=args.device)
        lat["images_per_s"] = bs * 1000.0 / lat["median"]
        energy = measure_energy(fwd, iters=args.iters, device=args.device)
        result[f"batch{bs}"] = {"latency": lat, "energy": energy}
    return result


def fmt(v, spec=".2f"):
    return "n/a" if v is None else format(v, spec)


def print_table(variants, batch_sizes):
    header = f"{'variant':<10} {'batch':>6} {'ms (med)':>10} {'images/s':>10} {'J/fwd':>8} {'speedup':>8}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for bs in batch_sizes:
        base_ms = variants["baseline"][f"batch{bs}"]["latency"]["median"]
        for name, r in variants.items():
            lat = r[f"batch{bs}"]["latency"]
            j = r[f"batch{bs}"]["energy"]["joules_per_iter"]
            print(f"{name:<10} {bs:>6} {lat['median']:>10.2f} "
                  f"{lat['images_per_s']:>10.1f} {fmt(j, '.2f'):>8} "
                  f"{base_ms / lat['median']:>7.2f}x")
        print()
    print("-" * len(header))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--pretrained", action="store_true",
                   help="Load pretrained weights (irrelevant for latency)")
    p.add_argument("--freeze_manifest", required=True)
    p.add_argument("--batch_sizes", default="1,8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda"
                          or torch.cuda.is_available() else "cpu")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    is_dinov2 = args.model.startswith("dinov2")
    if is_dinov2:
        from transformers import AutoModelForImageClassification
        from surgery_dinov2 import Dinov2FreezeHookManager, build_sliced_dinov2
        hf_id = ("facebook/dinov2-giant-imagenet1k-1-layer"
                 if args.model == "dinov2_giant" else args.model)
        print(f"Loading {hf_id} on {device} ...")
        model = AutoModelForImageClassification.from_pretrained(hf_id)
        input_size = 518
        hook_cls, slicer = Dinov2FreezeHookManager, build_sliced_dinov2
    else:
        import timm
        print(f"Loading {args.model} (pretrained={args.pretrained}) on {device} ...")
        model = timm.create_model(args.model, pretrained=args.pretrained)
        input_size = 224
        hook_cls, slicer = ViTFreezeHookManager, build_sliced_vit
    model.to(device)
    model.eval()

    with open(args.freeze_manifest) as f:
        manifest = json.load(f)
    frozen_heads = load_frozen_heads(manifest)
    print(f"Manifest: {len(frozen_heads)} frozen heads")

    variants = {}

    variants["baseline"] = bench_variant("baseline", model, batch_sizes, args,
                                         device, input_size)

    mgr = hook_cls(model, frozen_heads)
    mgr.register()
    variants["hooked"] = bench_variant("hooked", model, batch_sizes, args,
                                       device, input_size)
    mgr.remove()

    # In-place surgery: model is no longer the baseline after this point.
    slicer(model, manifest)
    variants["sliced"] = bench_variant("sliced", model, batch_sizes, args,
                                       device, input_size)

    out = {
        "config": {
            "model": args.model,
            "pretrained": args.pretrained,
            "freeze_manifest": args.freeze_manifest,
            "batch_sizes": batch_sizes,
            "device": str(device),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "frozen_heads": len(frozen_heads),
        },
        "variants": variants,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print_table(variants, batch_sizes)


if __name__ == "__main__":
    main()
