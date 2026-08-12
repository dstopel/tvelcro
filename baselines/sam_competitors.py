#!/usr/bin/env python3
"""Evaluate competitor SAM variants on the thesis segmentation domains.

Runs MobileSAM and SlimSAM under the exact T-VELCRO protocol: automatic mask
generation on the same eval images, Hungarian-matched fidelity against SAM
ViT-H reference masks, ground-truth pixel coverage, and measured latency.
This turns the quoted-numbers comparison in baselines/README.md into a
measured, same-dataset table.

Variants:
  mobile_sam   MobileSAM (ViT-T distilled encoder), drop-in segment-anything
               API. Requires: pip install git+https://github.com/ChaoningZhang/MobileSAM.git
               and the checkpoint at $DATA_ROOT/mobile_sam.pt.
  slimsam_50 / slimsam_77
               SlimSAM pruned checkpoints via the HuggingFace mask-generation
               pipeline (Zigeng/SlimSAM-uniform-50 / -77); weights auto-download.

EfficientSAM ships no automatic-mask-generation pipeline, so it stays a
quoted-numbers row (see baselines/README.md).

Reference masks are generated once with SAM ViT-H and cached under
<out>/reference/<domain>/, so adding variants does not recompute them.

Usage (from repo root, PYTHONPATH=.):
  python baselines/sam_competitors.py --variants mobile_sam slimsam_50 \
      --domains kvasir isic busi pcb deepglobe --max_images 50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam"))
from sam_velcro_pkg.dataset import ImageDirDataset          # noqa: E402
from sam_velcro_pkg.evaluation import evaluate_masks        # noqa: E402
from sam_velcro_pkg.gt_masks import load_gt_masks           # noqa: E402
from common.timing import measure_latency                   # noqa: E402

GT_DOMAINS = {"kvasir", "isic", "busi", "deepglobe", "pcb"}


def _to_amg_dicts(bool_masks):
    """Wrap plain bool masks in the AMG dict format evaluate_masks expects."""
    return [{"segmentation": m.astype(bool)} for m in bool_masks]


def _gt_coverage(gt_masks, pred_masks):
    """Fraction of GT pixels covered by the union of predicted masks."""
    if not gt_masks:
        return None
    gt_union = np.zeros_like(gt_masks[0], dtype=bool)
    for g in gt_masks:
        gt_union |= g.astype(bool)
    if gt_union.sum() == 0:
        return None
    if pred_masks:
        pred_union = np.zeros_like(gt_union)
        for p in pred_masks:
            seg = p["segmentation"] if isinstance(p, dict) else p
            if seg.shape != pred_union.shape:
                import cv2
                seg = cv2.resize(seg.astype(np.uint8),
                                 (pred_union.shape[1], pred_union.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            pred_union |= seg
    else:
        pred_union = np.zeros_like(gt_union)
    return float((gt_union & pred_union).sum() / gt_union.sum())


def build_reference(args, device):
    """SAM ViT-H automatic mask generator (the fidelity reference)."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.88, stability_score_thresh=0.95)
    n_params = sum(p.numel() for p in sam.image_encoder.parameters())
    return gen, n_params


def build_mobile_sam(args, device):
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    ckpt = Path(args.data_root) / "mobile_sam.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} missing. Download: "
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
    sam = sam_model_registry["vit_t"](checkpoint=str(ckpt)).to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.88, stability_score_thresh=0.95)
    n_params = sum(p.numel() for p in sam.image_encoder.parameters())

    def run(image_rgb):
        return gen.generate(image_rgb)
    return run, n_params


def build_slimsam(variant, args, device):
    from transformers import pipeline
    model_id = {"slimsam_50": "Zigeng/SlimSAM-uniform-50",
                "slimsam_77": "Zigeng/SlimSAM-uniform-77"}[variant]
    pipe = pipeline("mask-generation", model=model_id,
                    device=0 if device == "cuda" else -1,
                    points_per_batch=64)
    n_params = sum(p.numel() for p in pipe.model.vision_encoder.parameters())

    def run(image_rgb):
        from PIL import Image
        out = pipe(Image.fromarray(image_rgb),
                   points_per_crop=args.points_per_side ** 2)
        return _to_amg_dicts([np.array(m) for m in out["masks"]])
    return run, n_params


VARIANT_BUILDERS = {
    "mobile_sam": build_mobile_sam,
    "slimsam_50": lambda a, d: build_slimsam("slimsam_50", a, d),
    "slimsam_77": lambda a, d: build_slimsam("slimsam_77", a, d),
}


def reference_masks_for(args, gen, domain, dataset, cache_dir):
    """Generate (or load cached) SAM ViT-H reference masks per image."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    for idx in range(len(dataset)):
        img, name = dataset[idx]
        npz = cache_dir / (Path(name).stem + ".npz")
        if npz.exists():
            data = np.load(npz)
            refs[name] = _to_amg_dicts([data[k] for k in data.files])
        else:
            masks = gen.generate(img)
            refs[name] = masks
            np.savez_compressed(
                npz, *[m["segmentation"] for m in masks])
    return refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=["mobile_sam"],
                   choices=sorted(VARIANT_BUILDERS))
    p.add_argument("--domains", nargs="+",
                   default=["kvasir", "isic", "busi", "pcb", "deepglobe"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_root", default=None,
                   help="default: <data_root>/image_dirs")
    p.add_argument("--sam_ckpt", default="./data/sam_vit_h_4b8939.pth")
    p.add_argument("--max_images", type=int, default=50)
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--latency_images", type=int, default=5)
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--out", default="./results/baselines/sam_competitors")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_root = Path(args.image_root or (Path(args.data_root) / "image_dirs"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  domains={args.domains}  variants={args.variants}")
    ref_gen, ref_params = build_reference(args, device)
    results = {"reference": {"name": "SAM ViT-H", "encoder_params": ref_params},
               "variants": {}}

    datasets = {}
    references = {}
    for domain in args.domains:
        img_dir = image_root / f"{domain}_eval"
        if not img_dir.exists():
            img_dir = image_root / domain
        datasets[domain] = ImageDirDataset(str(img_dir), domain,
                                           max_images=args.max_images)
        references[domain] = reference_masks_for(
            args, ref_gen, domain, datasets[domain],
            out_dir / "reference" / domain)
        print(f"[ref] {domain}: {len(datasets[domain])} images")

    for variant in args.variants:
        try:
            run_fn, n_params = VARIANT_BUILDERS[variant](args, device)
        except Exception as exc:  # missing package/checkpoint disables one variant
            print(f"[skip] {variant}: {exc}")
            results["variants"][variant] = {"error": str(exc)}
            continue
        vres = {"encoder_params": n_params, "domains": {}}
        for domain in args.domains:
            ds = datasets[domain]
            per_img = []
            for idx in range(len(ds)):
                img, name = ds[idx]
                pred = run_fn(img)
                ev = evaluate_masks(references[domain][name], pred)
                row = {"image": name, "mean_iou": ev.get("mean_iou"),
                       "miss_rate": ev.get("miss_rate"),
                       "false_pos_rate": ev.get("false_pos_rate")}
                if domain in GT_DOMAINS:
                    try:
                        gt = load_gt_masks(name, domain, args.data_root)
                        row["gt_coverage"] = _gt_coverage(gt, pred)
                    except Exception:
                        row["gt_coverage"] = None
                per_img.append(row)
            agg = {}
            for k in ("mean_iou", "miss_rate", "false_pos_rate", "gt_coverage"):
                vals = [r[k] for r in per_img if r.get(k) is not None]
                agg[k] = float(np.mean(vals)) if vals else None
            agg["n_images"] = len(per_img)
            if not args.skip_latency and len(ds) > 0:
                imgs = [ds[i][0] for i in range(min(args.latency_images, len(ds)))]
                lat = measure_latency(
                    lambda: [run_fn(im) for im in imgs], warmup=1, iters=3)
                agg["amg_ms_per_image"] = lat["median"] / max(1, len(imgs))
            vres["domains"][domain] = agg
            print(f"[{variant}] {domain}: IoU-vs-SAM={agg['mean_iou']:.4f} "
                  f"miss={agg['miss_rate']:.3f} cov={agg.get('gt_coverage')}")
        results["variants"][variant] = vres

    out_json = out_dir / "sam_competitors.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")
    # ponytail: table rendering lives in the paper scripts, JSON is the contract


if __name__ == "__main__":
    main()
