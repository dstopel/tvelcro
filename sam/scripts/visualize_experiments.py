#!/usr/bin/env python3
"""
visualize_experiments.py — Generate multi-panel comparison figures.

Produces two types of figures per dataset:

Figure 1 — Attention-only sweep:
  [Original] [12.5% attn] [25% attn] [37.5% attn]

Figure 2 — Combined (attn 25% fixed + MLP sweep):
  [Original] [+MLP 12.5%] [+MLP 25%] [+MLP 37.5%]

Each panel shows colored mask overlay with GT contour in green.

Usage:
  python3 scripts/visualize_experiments.py \
      --checkpoint ./data/sam_vit_h_4b8939.pth \
      --domain kvasir \
      --image_dir ./image_dirs_cross_domain \
      --data_root ./data \
      --out_dir ./visualizations
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_velcro_pkg.dataset import ImageDirDataset
from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
from sam_velcro_pkg.hooks import FreezeHookManager
from sam_velcro_pkg.gt_masks import load_gt_masks
from sam_velcro_pkg.mlp_hooks import MLPFreezeHookManager, select_mlp_neurons_to_freeze


DOMAIN_OBJECTS = {
    "kvasir": "polyps",
    "isic": "lesions",
    "busi": "tumors",
    "pcb": "defects",
}

ATTN_FRACTIONS = [0.125, 0.25, 0.375]
MLP_FRACTIONS = [0.125, 0.25, 0.375]


def overlay_masks(image: np.ndarray, masks: List[dict], alpha: float = 0.4) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    np.random.seed(42)
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    for mask in sorted_masks:
        seg = mask["segmentation"]
        color = np.random.randint(60, 255, size=3).astype(np.float32)
        overlay[seg] = overlay[seg] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)


def overlay_gt_contour(image: np.ndarray, gt_masks: List[np.ndarray]) -> np.ndarray:
    result = image.copy()
    for gt in gt_masks:
        contours, _ = cv2.findContours(
            gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def find_attn_manifest(domain: str, frac: float) -> Optional[str]:
    candidates = [
        f"./results_cross_domain/{domain}/freeze_{domain}_f{frac}/freeze_manifest.json",
        f"./results_cross_domain/freeze_{domain}_f{frac}/freeze_manifest.json",
        f"./results_wave2/{domain}/freeze_{domain}_f{frac}/freeze_manifest.json",
        f"./results_wave2/freeze_{domain}_f{frac}/freeze_manifest.json",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def generate_masks_with_config(
    gen: SamAutomaticMaskGenerator,
    sam,
    image: np.ndarray,
    device: torch.device,
    attn_manifest_path: Optional[str] = None,
    mlp_profile_dir: Optional[str] = None,
    mlp_fraction: Optional[float] = None,
    domain: Optional[str] = None,
) -> List[dict]:
    """Generate masks with optional attention and/or MLP freezing."""

    attn_mgr = None
    mlp_mgr = None

    if attn_manifest_path:
        manifest = load_manifest(Path(attn_manifest_path))
        attn_frozen = load_frozen_heads_from_manifest(manifest)
        attn_mgr = FreezeHookManager(sam, attn_frozen, device)

    if mlp_profile_dir and mlp_fraction and domain:
        mlp_frozen = select_mlp_neurons_to_freeze(
            Path(mlp_profile_dir), domain, mlp_fraction
        )
        if mlp_frozen:
            mlp_mgr = MLPFreezeHookManager(sam, mlp_frozen, device)

    with torch.inference_mode():
        if attn_mgr:
            attn_mgr.register()
        if mlp_mgr:
            mlp_mgr.register()

        masks = gen.generate(image)

        if attn_mgr:
            attn_mgr.remove()
        if mlp_mgr:
            mlp_mgr.remove()

    return masks


def make_panel(image, masks, gt_masks, title) -> Tuple[np.ndarray, str]:
    overlay = overlay_masks(image, masks)
    if gt_masks:
        overlay = overlay_gt_contour(overlay, gt_masks)
    return overlay, title


def save_figure(panels: List[Tuple[np.ndarray, str]], suptitle: str, out_path: Path):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=13, y=0.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    domain = args.domain
    objects_name = DOMAIN_OBJECTS.get(domain, "objects")

    # Load model
    print(f"Loading SAM ViT-H...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95
    )

    # Load image
    eval_dir = f"{args.image_dir}/{domain}_eval"
    dataset = ImageDirDataset(eval_dir, domain)
    idx = min(args.image_index, len(dataset) - 1)
    image, _, filename = dataset[idx]
    print(f"Image: {filename} (index {idx})", flush=True)

    # Load GT
    gt_masks = load_gt_masks(filename, domain, args.data_root)
    gt_info = f"GT: {len(gt_masks)} {objects_name}" if gt_masks else "No GT"

    # =========================================================================
    # Figure 1: Attention-only sweep
    # =========================================================================
    print(f"\n[Figure 1] Attention-only sweep...", flush=True)

    # Original (no freeze)
    print(f"  Generating original masks...", flush=True)
    with torch.inference_mode():
        masks_orig = gen.generate(image)
    panels = [make_panel(image, masks_orig, gt_masks,
                         f"Original\n{len(masks_orig)} masks")]

    for frac in ATTN_FRACTIONS:
        manifest_path = find_attn_manifest(domain, frac)
        if not manifest_path:
            print(f"  SKIP attn {frac}: no manifest found", flush=True)
            continue
        print(f"  Generating attn {frac*100:.1f}% masks...", flush=True)
        masks = generate_masks_with_config(gen, sam, image, device,
                                           attn_manifest_path=manifest_path)
        n_miss = len(masks_orig) - len(masks) if len(masks) < len(masks_orig) else 0
        panels.append(make_panel(
            image, masks, gt_masks,
            f"Attn {frac*100:.0f}%\n{len(masks)} masks"
        ))

    out_path = Path(args.out_dir) / f"{domain}_attention_sweep.png"
    save_figure(panels, f"{domain.upper()} — Attention Head Freeze Sweep | {filename} | {gt_info}",
                out_path)

    # =========================================================================
    # Figure 2: Combined (attn 25% + MLP sweep)
    # =========================================================================
    print(f"\n[Figure 2] Combined attn 25% + MLP sweep...", flush=True)

    attn_manifest_25 = find_attn_manifest(domain, 0.25)
    mlp_profile_dir = f"./results_combined_freeze/{domain}/mlp_profile"

    if not attn_manifest_25:
        print(f"  SKIP: no 25% attention manifest for {domain}", flush=True)
    elif not Path(mlp_profile_dir).exists():
        print(f"  SKIP: no MLP profile at {mlp_profile_dir}", flush=True)
        print(f"  Run the combined experiment first.", flush=True)
    else:
        panels2 = [make_panel(image, masks_orig, gt_masks,
                              f"Original\n{len(masks_orig)} masks")]

        for mlp_frac in MLP_FRACTIONS:
            print(f"  Generating attn 25% + MLP {mlp_frac*100:.1f}%...", flush=True)
            masks = generate_masks_with_config(
                gen, sam, image, device,
                attn_manifest_path=attn_manifest_25,
                mlp_profile_dir=mlp_profile_dir,
                mlp_fraction=mlp_frac,
                domain=domain,
            )
            panels2.append(make_panel(
                image, masks, gt_masks,
                f"Attn 25% + MLP {mlp_frac*100:.0f}%\n{len(masks)} masks"
            ))

        out_path2 = Path(args.out_dir) / f"{domain}_combined_sweep.png"
        save_figure(panels2,
                    f"{domain.upper()} — Combined Freeze (Attn 25% + MLP sweep) | {filename} | {gt_info}",
                    out_path2)

    # Cleanup
    del sam, gen
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-panel freeze comparison figures"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_OBJECTS.keys()))
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="./visualizations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_index", type=int, default=5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
