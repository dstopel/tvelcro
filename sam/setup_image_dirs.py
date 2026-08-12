#!/usr/bin/env python3
"""
setup_image_dirs.py

Creates PTQ4ViT-style image directories from COCO val2017:
  {group}/             — all images for that domain
  {group}_profile/     — 300 images for profiling (first half)
  {group}_eval/        — 300 images for evaluation (second half, non-overlapping)

Images are symlinked (not copied) to save disk space.

Usage:
    python setup_image_dirs.py --coco_dir ./data --out_dir ./image_dirs --n_profile 300 --n_eval 300
"""

import argparse
import json
import os
import random
from pathlib import Path

from sam_velcro_pkg.constants import DOMAIN_GROUPS


def main():
    parser = argparse.ArgumentParser(description="Setup image directories for SAM VELCRO")
    parser.add_argument("--coco_dir", required=True, help="COCO root (must contain val2017/ and annotations/)")
    parser.add_argument("--out_dir", default="./image_dirs", help="Output directory for image dirs")
    parser.add_argument("--n_profile", type=int, default=300, help="Images per group for profiling")
    parser.add_argument("--n_eval", type=int, default=300, help="Images per group for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir)
    img_dir = coco_dir / "val2017"
    ann_path = coco_dir / "annotations" / "instances_val2017.json"

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found. Run download_data.sh first.")
        return
    if not ann_path.exists():
        print(f"ERROR: {ann_path} not found. Run download_data.sh first.")
        return

    print("Loading COCO annotations...")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    all_image_ids = [img["id"] for img in coco["images"]]

    cat_to_images = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in cat_to_images:
            cat_to_images[cid] = set()
        cat_to_images[cid].add(ann["image_id"])

    random.seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir_abs = img_dir.resolve()

    print(f"\nCreating image directories in {out_dir}/")
    print(f"  Profile: {args.n_profile} images, Eval: {args.n_eval} images per group")
    print(f"  Seed: {args.seed}")
    print(f"{'='*60}")

    for group_name, categories in DOMAIN_GROUPS.items():
        if categories is None:
            image_ids = list(all_image_ids)
        else:
            target_cat_ids = {cat_name_to_id[n] for n in categories if n in cat_name_to_id}
            image_ids = set()
            for cid in target_cat_ids:
                if cid in cat_to_images:
                    image_ids.update(cat_to_images[cid])
            image_ids = sorted(image_ids)

        random.shuffle(image_ids)

        n_total = len(image_ids)
        n_profile = min(args.n_profile, n_total // 2)
        n_eval = min(args.n_eval, n_total - n_profile)

        profile_ids = image_ids[:n_profile]
        eval_ids = image_ids[n_profile:n_profile + n_eval]
        all_group_ids = image_ids[:n_profile + n_eval]

        dirs = {
            group_name: all_group_ids,
            f"{group_name}_profile": profile_ids,
            f"{group_name}_eval": eval_ids,
        }

        for dir_name, ids in dirs.items():
            target_dir = out_dir / dir_name / "images"
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_id in ids:
                fname = id_to_filename[img_id]
                src = img_dir_abs / fname
                dst = target_dir / fname
                if not dst.exists():
                    os.symlink(src, dst)

        cats_str = ", ".join(categories) if categories else "ALL"
        print(f"  {group_name:<14} | total={n_total:>5} | profile={n_profile:>4} | eval={n_eval:>4} | categories: {cats_str}")

    print(f"\n{'='*60}")
    print(f"Done! Image directories created at: {out_dir}/")
    print(f"\nDirectory structure:")

    for group_name in DOMAIN_GROUPS:
        for suffix in ["", "_profile", "_eval"]:
            d = out_dir / f"{group_name}{suffix}" / "images"
            if d.exists():
                count = len(list(d.iterdir()))
                print(f"  {group_name}{suffix}/images/ ({count} images)")


if __name__ == "__main__":
    main()
