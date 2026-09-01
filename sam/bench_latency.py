"""Latency and energy benchmark: baseline vs hooked vs sliced SAM ViT-H.

Measures, per variant:
  - encoder-only forward (1x3x1024x1024) wall-clock stats and energy
  - full SamAutomaticMaskGenerator per image over --eval_n images

Run from repo root with PYTHONPATH=. (see experiments/sam/exp_structured_latency.sh):
  PYTHONPATH=. python3 sam/bench_latency.py --checkpoint ... --freeze_manifest ...

Note: the sliced variant mutates the loaded model in place, so it always runs last.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.timing import EnergyMeter, measure_energy, measure_latency
from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
from sam_velcro_pkg.hooks import FreezeHookManager
from sam_velcro_pkg.utils import load_sam_model
from surgery import build_sliced_sam


def bench_variant(name, sam, images, args, use_cuda):
    from segment_anything import SamAutomaticMaskGenerator

    result = {"encoder_params": sum(p.numel() for p in sam.image_encoder.parameters()),
              "dtype": str(next(sam.parameters()).dtype)}
    x = torch.randn(1, 3, 1024, 1024, device=next(sam.parameters()).device)

    def enc_fwd():
        sam.image_encoder(x)

    print(f"[{name}] encoder-only forward ...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    result["encoder"] = measure_latency(enc_fwd, warmup=args.warmup,
                                        iters=args.iters, device=args.device)
    result["encoder_energy"] = measure_energy(enc_fwd, iters=args.iters,
                                              device=args.device)
    if use_cuda:
        result["encoder_peak_mem_bytes"] = torch.cuda.max_memory_allocated()

    if images:
        print(f"[{name}] full AutomaticMaskGenerator over {len(images)} images ...")
        gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=args.points_per_side,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        # Warmup one image, then time each image with CUDA fences, energy over
        # the whole timed loop.
        with torch.inference_mode():
            gen.generate(images[0])
        per_image_s = []
        if use_cuda:
            torch.cuda.synchronize()
        with EnergyMeter() as em:
            with torch.inference_mode():
                for img in images:
                    if use_cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    gen.generate(img)
                    if use_cuda:
                        torch.cuda.synchronize()
                    per_image_s.append(time.perf_counter() - t0)
        arr = np.asarray(per_image_s)
        e = em.as_dict()
        result["maskgen"] = {
            "median_s_per_image": float(np.median(arr)),
            "mean_s_per_image": float(arr.mean()),
            "std_s_per_image": float(arr.std()),
            "num_images": len(images),
        }
        result["maskgen_energy"] = dict(
            e, joules_per_image=None if e["joules"] is None else e["joules"] / len(images))
    return result


def fmt(v, spec=".1f"):
    return "n/a" if v is None else format(v, spec)


def print_table(variants):
    base_enc = variants["baseline"]["encoder"]["median"]
    base_gen = variants["baseline"].get("maskgen", {}).get("median_s_per_image")
    header = (f"{'variant':<10} {'enc ms (med)':>13} {'enc J/fwd':>10} "
              f"{'gen s/img':>10} {'gen J/img':>10} {'enc speedup':>12}")
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in variants.items():
        enc_ms = r["encoder"]["median"]
        enc_j = r["encoder_energy"]["joules_per_iter"]
        gen_s = r.get("maskgen", {}).get("median_s_per_image")
        gen_j = r.get("maskgen_energy", {}).get("joules_per_image")
        speedup = base_enc / enc_ms if enc_ms else None
        print(f"{name:<10} {fmt(enc_ms, '.1f'):>13} {fmt(enc_j, '.1f'):>10} "
              f"{fmt(gen_s, '.2f'):>10} {fmt(gen_j, '.1f'):>10} "
              f"{fmt(speedup, '.2f'):>11}x")
    if base_gen:
        print(f"(gen speedup sliced vs baseline: "
              f"{fmt(base_gen / variants['sliced']['maskgen']['median_s_per_image'], '.2f')}x)")
    print("-" * len(header))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--freeze_manifest", required=True)
    p.add_argument("--image_dir", default=None,
                   help="Directory of eval images for the mask-generator pass")
    p.add_argument("--eval_n", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--points_per_side", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--mlp_profile_dir", default=None,
                   help="Enable the combined config: MLP profile root dir")
    p.add_argument("--mlp_group", default=None,
                   help="Domain group name inside the MLP profile dir")
    p.add_argument("--mlp_fraction", type=float, default=0.0,
                   help="Fraction of MLP neurons to freeze (combined config)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda"
                          or torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    print(f"Loading SAM ViT-H from {args.checkpoint} on {device} ...")
    sam = load_sam_model(args.checkpoint, device)
    sam.eval()

    manifest = load_manifest(Path(args.freeze_manifest))
    frozen_heads = load_frozen_heads_from_manifest(manifest)
    print(f"Manifest: {len(frozen_heads)} frozen heads "
          f"({manifest['summary']['frozen_windowed']} windowed, "
          f"{manifest['summary']['frozen_global']} global)")

    images = []
    if args.image_dir and Path(args.image_dir).is_dir():
        from sam_velcro_pkg.dataset import ImageDirDataset
        ds = ImageDirDataset(args.image_dir, "bench", max_images=args.eval_n)
        images = [ds[i][0] for i in range(len(ds))]
        print(f"Loaded {len(images)} eval images from {args.image_dir}")
    else:
        print(f"WARNING: image_dir missing ({args.image_dir}); "
              "skipping mask-generator benchmark, encoder-only results only.")

    frozen_neurons = {}
    if args.mlp_profile_dir and args.mlp_group and args.mlp_fraction > 0:
        from sam_velcro_pkg.mlp_hooks import select_mlp_neurons_to_freeze
        frozen_neurons = select_mlp_neurons_to_freeze(
            Path(args.mlp_profile_dir), args.mlp_group, args.mlp_fraction)
        print(f"Combined config: {len(frozen_neurons)} frozen MLP neurons "
              f"({args.mlp_fraction:.1%})")

    variants = {}

    variants["baseline"] = bench_variant("baseline", sam, images, args, use_cuda)

    mgr = FreezeHookManager(sam, frozen_heads, device)
    mgr.register()
    mlp_mgr = None
    if frozen_neurons:
        from sam_velcro_pkg.mlp_hooks import MLPFreezeHookManager
        mlp_mgr = MLPFreezeHookManager(sam, frozen_neurons, device)
        mlp_mgr.register()
    variants["hooked"] = bench_variant("hooked", sam, images, args, use_cuda)
    mgr.remove()
    if mlp_mgr is not None:
        mlp_mgr.remove()

    # In-place surgery: model is no longer the baseline after this point.
    build_sliced_sam(sam, manifest)
    if frozen_neurons:
        from surgery_mlp import build_sliced_sam_mlp
        build_sliced_sam_mlp(sam, frozen_neurons)
    variants["sliced"] = bench_variant("sliced", sam, images, args, use_cuda)

    out = {
        "config": {
            "checkpoint": args.checkpoint,
            "freeze_manifest": args.freeze_manifest,
            "image_dir": args.image_dir,
            "eval_n": len(images),
            "device": str(device),
            "points_per_side": args.points_per_side,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "frozen_heads": len(frozen_heads),
            "frozen_mlp_neurons": len(frozen_neurons),
            "mlp_fraction": args.mlp_fraction,
        },
        "variants": variants,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print_table(variants)


if __name__ == "__main__":
    main()
