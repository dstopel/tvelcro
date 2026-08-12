#!/usr/bin/env python3
"""
sam_velcro.py — Value Locality Compression for SAM ViT-H

Three-phase pipeline:

Phase 1 (profile):  Welford variance + mean per attention head across images.
Phase 2 (freeze):   S-VELCRO channel-rank: rank heads by variance sum,
                     freeze the T_k lowest-ranked heads per block.
Phase 3 (eval):     Replace frozen heads with precomputed means, compare
                     masks from original vs patched model at 4 levels:
                       L1 — Activation cosine similarity (per-block)
                       L2 — Encoder feature comparison (64×64×256)
                       L3 — Mask IoU via Hungarian matching
                       L4 — Visual side-by-side overlays

Architecture:
  SAM ViT-H: 32 blocks × 16 heads × 80 head_dim × 1280 hidden
  28 windowed blocks (14×14 window, 196 tokens each)
  4 global blocks (blocks 7, 15, 23, 31, full 4096 tokens)
  Hook target: image_encoder.blocks[i].attn.proj INPUT (pre-projection)
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

from sam_velcro_pkg.constants import AVAILABLE_MODES, DOMAIN_GROUPS, NUM_BLOCKS, NUM_HEADS
from sam_velcro_pkg.utils import Timer, get_device, load_sam_model, save_json, setup_logging


# ============================================================================
# Phase 1: Profile
# ============================================================================

def run_profile(args):
    logger = setup_logging(Path(args.out_dir), "profile")
    device = get_device(args.device)

    logger.info("Loading SAM ViT-H...")
    sam = load_sam_model(args.checkpoint, device)

    groups = [g.strip() for g in args.domain_groups.split(",")]
    logger.info(f"Domain groups: {groups}")

    from sam_velcro_pkg.dataset import COCODomainDataset, ImageDirDataset
    from sam_velcro_pkg.hooks import ProfileHookManager
    from sam_velcro_pkg.plotting import plot_variance_heatmap

    for group_name in groups:
        logger.info(f"\n{'='*60}")
        logger.info(f"Profiling group: {group_name}")

        if args.image_dir:
            img_dir = Path(args.image_dir) / f"{group_name}_profile"
            if not img_dir.exists():
                img_dir = Path(args.image_dir) / group_name
            dataset = ImageDirDataset(str(img_dir), group_name, args.max_images)
        else:
            categories = DOMAIN_GROUPS.get(group_name)
            dataset = COCODomainDataset(args.coco_dir, group_name, categories, args.max_images)

        logger.info(f"  Images: {len(dataset)}")

        hook_mgr = ProfileHookManager(sam, device)
        hook_mgr.register()

        mlp_mgr = None
        if args.profile_mlp:
            from sam_velcro_pkg.mlp_hooks import MLPProfileHookManager
            mlp_mgr = MLPProfileHookManager(sam, device)
            mlp_mgr.register()

        with torch.inference_mode():
            for idx in range(len(dataset)):
                image, img_id, filename = dataset[idx]
                sam.image_encoder(sam.preprocess(
                    torch.as_tensor(image, device=device)
                    .permute(2, 0, 1).unsqueeze(0).float()
                ))
                hook_mgr.increment_sample_count()
                if mlp_mgr is not None:
                    mlp_mgr.increment_sample_count()

                if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
                    logger.info(f"  [{group_name}] {idx+1}/{len(dataset)} images profiled")

        summary = hook_mgr.save(Path(args.out_dir), group_name)
        save_json(summary, Path(args.out_dir) / group_name / "profile_summary.json")

        if mlp_mgr is not None:
            mlp_summary = mlp_mgr.save(Path(args.out_dir), group_name)
            save_json(mlp_summary, Path(args.out_dir) / group_name / "mlp_profile_summary.json")
            mlp_mgr.remove()
            logger.info(f"  Saved MLP profiles for {len(mlp_summary['neurons'])} blocks")

        plot_variance_heatmap(
            Path(args.out_dir), group_name,
            Path(args.out_dir) / group_name / "plots" / "variance_heatmap.png",
        )

        hook_mgr.remove()
        logger.info(f"  Saved {len(summary['heads'])} head profiles to {args.out_dir}/{group_name}/")

    logger.info("\nProfile phase complete.")

    del sam
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Phase 2: Freeze
# ============================================================================

def run_freeze(args):
    logger = setup_logging(Path(args.out_dir), "freeze")

    from sam_velcro_pkg.freeze import (
        build_manifest,
        compute_channel_ranks,
        portable_profile_root,
        save_manifest,
        select_heads_by_ratio,
        select_heads_to_freeze,
    )
    from sam_velcro_pkg.plotting import plot_freeze_heatmap, plot_rank_scores_per_block

    logger.info(f"Loading profile from: {args.profile_dir}/{args.profile_group}")

    entries = compute_channel_ranks(Path(args.profile_dir), args.profile_group)

    # Optional MLP freeze config, recorded in the manifest for eval to pick up
    mlp_cfg = None
    if args.mlp_fraction is not None:
        mlp_root = args.mlp_profile_dir or args.profile_dir
        mlp_group = args.mlp_profile_group or args.profile_group
        mlp_cfg = {
            "fraction": args.mlp_fraction,
            "profile_root": portable_profile_root(mlp_root),
            "profile_group": mlp_group,
        }
        if not (Path(mlp_root) / mlp_group / "block0_mlp_variance.npy").exists():
            logger.warning(f"  No MLP profiles found at {mlp_root}/{mlp_group}/ "
                           f"(profile with --profile_mlp first)")

    if args.freeze_method == "ratio":
        logger.info(f"  Freeze method: ratio (threshold={args.ratio_threshold}, general={args.general_group})")
        frozen = select_heads_by_ratio(
            Path(args.profile_dir), args.profile_group, args.general_group, args.ratio_threshold,
        )
        logger.info(f"  Freezing {len(frozen)} / {NUM_BLOCKS * NUM_HEADS} heads (ratio > {args.ratio_threshold})")
        manifest = build_manifest(
            frozen, args.profile_group, None, None, ratio_threshold=args.ratio_threshold,
            profile_root=args.profile_dir, mlp=mlp_cfg,
        )
    else:
        ranking = "variance" if args.freeze_method == "rank" else args.freeze_method
        logger.info(f"  Found {len(entries)} head profiles")
        logger.info(f"  Ranking method: {ranking} (seed={args.seed})")
        frozen = select_heads_to_freeze(
            entries,
            freeze_fraction=args.freeze_fraction,
            num_freeze_per_block=args.num_freeze_per_block,
            ranking_method=ranking,
            seed=args.seed,
        )
        logger.info(f"  Freezing {len(frozen)} / {NUM_BLOCKS * NUM_HEADS} heads")
        manifest = build_manifest(
            frozen, args.profile_group, args.freeze_fraction, args.num_freeze_per_block,
            profile_root=args.profile_dir, ranking_method=ranking, seed=args.seed,
            mlp=mlp_cfg,
        )
    manifest_path = save_manifest(manifest, Path(args.out_dir))
    logger.info(f"  Manifest saved to {manifest_path}")

    frozen_set = {(e["block"], e["head"]) for e in frozen}

    plot_freeze_heatmap(
        frozen_set,
        Path(args.out_dir) / "plots" / "freeze_heatmap.png",
        title=f"Frozen Heads — {args.profile_group} (frac={args.freeze_fraction})",
    )

    plot_rank_scores_per_block(
        entries, frozen_set,
        Path(args.out_dir) / "plots" / "rank_bar_charts",
    )

    logger.info(f"  Summary: {manifest['summary']}")
    logger.info("Freeze phase complete.")


# ============================================================================
# Phase 3: Eval
# ============================================================================

def run_eval(args):
    logger = setup_logging(Path(args.out_dir), "eval")
    device = get_device(args.device)

    from segment_anything import SamAutomaticMaskGenerator

    from sam_velcro_pkg.dataset import COCODomainDataset, ImageDirDataset
    from sam_velcro_pkg.evaluation import (
        DOMAIN_THRESHOLDS,
        MultiLevelEvaluator,
        classify_misses,
        compare_encoder_features,
        evaluate_masks,
    )
    from sam_velcro_pkg.freeze import load_frozen_heads_from_manifest, load_manifest
    from sam_velcro_pkg.gt_masks import load_gt_masks
    from sam_velcro_pkg.hooks import FreezeHookManager
    from sam_velcro_pkg.plotting import (
        plot_cosine_sim_curve,
        plot_feature_cosine_map,
        plot_iou_distribution,
        plot_mask_overlay_comparison,
    )

    logger.info("Loading SAM ViT-H...")
    sam = load_sam_model(args.checkpoint, device)

    logger.info(f"Loading freeze manifest: {args.freeze_manifest}")
    manifest = load_manifest(Path(args.freeze_manifest))
    frozen_heads = load_frozen_heads_from_manifest(manifest)
    logger.info(f"  Frozen heads: {len(frozen_heads)} / {NUM_BLOCKS * NUM_HEADS}")

    eval_groups = list(dict.fromkeys(g.strip() for g in args.eval_groups.split(",")))

    collect_diag = args.diagnostic_level >= 1
    freeze_mgr = FreezeHookManager(sam, frozen_heads, device, collect_diagnostics=collect_diag)

    # Optional MLP freeze: CLI flags override the manifest's recorded mlp config
    manifest_mlp = manifest.get("mlp") or {}
    mlp_fraction = args.mlp_fraction if args.mlp_fraction is not None else manifest_mlp.get("fraction")
    mlp_mgr = None
    if mlp_fraction:
        from sam_velcro_pkg.mlp_hooks import MLPFreezeHookManager, select_mlp_neurons_to_freeze
        mlp_root = args.mlp_profile_dir or manifest_mlp.get("profile_root")
        mlp_group = args.mlp_profile_group or manifest_mlp.get("profile_group")
        if not mlp_root or not mlp_group:
            print("ERROR: --mlp_profile_dir and --mlp_profile_group required "
                  "(freeze manifest has no mlp config)")
            sys.exit(1)
        mlp_frozen = select_mlp_neurons_to_freeze(Path(mlp_root), mlp_group, mlp_fraction)
        if not mlp_frozen:
            print(f"ERROR: no MLP profiles found at {mlp_root}/{mlp_group}/ "
                  f"(profile with --profile_mlp first)")
            sys.exit(1)
        logger.info(f"  MLP: freezing {len(mlp_frozen)} neurons "
                    f"(fraction={mlp_fraction}, profile={mlp_root}/{mlp_group})")
        mlp_mgr = MLPFreezeHookManager(sam, mlp_frozen, device)

    mask_gen_kwargs = dict(
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
    )

    for group_name in eval_groups:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating group: {group_name}")

        if args.image_dir:
            img_dir = Path(args.image_dir) / f"{group_name}_eval"
            if not img_dir.exists():
                img_dir = Path(args.image_dir) / group_name
            dataset = ImageDirDataset(str(img_dir), group_name, args.max_images)
        else:
            categories = DOMAIN_GROUPS.get(group_name)
            dataset = COCODomainDataset(args.coco_dir, group_name, categories, args.max_images)
        logger.info(f"  Images: {len(dataset)}")

        evaluator = MultiLevelEvaluator(Path(args.out_dir) / group_name)
        all_ious = []

        gen = SamAutomaticMaskGenerator(sam, **mask_gen_kwargs)

        skipped = 0
        for idx in range(len(dataset)):
            image, img_id, filename = dataset[idx]

            try:
                # --- Original pass (no hooks) ---
                with torch.inference_mode():
                    masks_orig = gen.generate(image)

                feat_orig = None
                if args.diagnostic_level >= 2:
                    with torch.inference_mode():
                        feat_orig = compare_encoder_features(sam, image, device)["features"]

                # --- Patched pass (with freeze hooks) ---
                freeze_mgr.register()
                if mlp_mgr is not None:
                    mlp_mgr.register()
                with torch.inference_mode():
                    masks_patch = gen.generate(image)

                feat_patch_result = None
                if args.diagnostic_level >= 2 and feat_orig is not None:
                    with torch.inference_mode():
                        feat_patch_result = compare_encoder_features(
                            sam, image, device, frozen_features=feat_orig,
                        )
                        evaluator.add_feature_similarity(feat_patch_result["mean_cosine_similarity"])

                if mlp_mgr is not None:
                    mlp_mgr.remove()
                freeze_mgr.remove()

                # --- Compare masks ---
                mask_result = evaluate_masks(masks_orig, masks_patch)
                evaluator.add_mask_result(mask_result)
                all_ious.extend([m[2] for m in mask_result["matched_pairs"]])

                # --- Miss / false-positive decomposition vs GT (when available) ---
                if group_name in DOMAIN_THRESHOLDS:
                    gt_masks = load_gt_masks(filename, group_name, args.data_root)
                    evaluator.add_miss_classification(classify_misses(
                        masks_orig, masks_patch, gt_masks, group_name,
                        matched_pairs=mask_result["matched_pairs"],
                    ))

                # --- Visualizations (first 10 images only) ---
                if args.save_visualizations and idx < 10:
                    plot_mask_overlay_comparison(
                        image, masks_orig, masks_patch,
                        Path(args.out_dir) / group_name / "level4_visual" / f"masks_{idx}.png",
                    )
                    if feat_patch_result is not None:
                        plot_feature_cosine_map(
                            feat_patch_result["cosine_map"],
                            Path(args.out_dir) / group_name / "level2_feature" / f"cosine_map_{idx}.png",
                            title=f"Feature Similarity — Image {idx}",
                        )

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if mlp_mgr is not None:
                    mlp_mgr.remove()
                freeze_mgr.remove()
                skipped += 1
                logger.warning(f"    SKIP image {idx} ({filename}): {type(e).__name__}: {str(e)[:100]}")

            # Explicit cleanup - keep memory flat
            gc.collect()
            torch.cuda.empty_cache()

            if (idx + 1) % 20 == 0:
                running_iou = np.mean(all_ious) if all_ious else 0
                logger.info(f"    {idx+1}/{len(dataset)} | running mean IoU: {running_iou:.4f}")

        if skipped:
            logger.info(f"  Skipped {skipped}/{len(dataset)} images due to OOM/size errors")

        # --- Aggregate and save results ---
        agg = evaluator.aggregate()
        logger.info(f"\n  Results for '{group_name}':")
        logger.info(f"    Mean IoU:       {agg.get('per_image_mean_iou', 0):.4f}")
        logger.info(f"    Miss rate:      {agg.get('per_image_miss_rate', 0):.4f}")
        logger.info(f"    False pos rate: {agg.get('per_image_false_pos_rate', 0):.4f}")
        if "mean_feature_cosine_sim" in agg:
            logger.info(f"    Feature sim:    {agg['mean_feature_cosine_sim']:.4f}")
        if "miss_decomposition" in agg:
            md = agg["miss_decomposition"]
            ad = agg["added_decomposition"]
            logger.info(f"    Misses: {md['total_misses']} "
                        f"(domain={md['domain_misses']}, redundant={md['redundant_misses']}, "
                        f"background={md['background_misses']})")
            logger.info(f"    Added:  {ad['total_added']} "
                        f"(domain={ad['added_domain']}, redundant={ad['added_redundant']}, "
                        f"background={ad['added_background']})")

        save_json(agg, Path(args.out_dir) / group_name / "eval_results.json")

        if all_ious:
            plot_iou_distribution(
                all_ious,
                Path(args.out_dir) / group_name / "level3_masks" / "iou_distribution.png",
            )

        if collect_diag:
            diag_summary = freeze_mgr.get_diagnostics_summary()
            save_json(diag_summary, Path(args.out_dir) / group_name / "level1_activation" / "diagnostics.json")
            plot_cosine_sim_curve(
                diag_summary,
                Path(args.out_dir) / group_name / "level1_activation" / "cosine_sim_curve.png",
            )
            freeze_mgr.reset_diagnostics()

    logger.info("\nEval phase complete.")

    del sam, gen, freeze_mgr, frozen_heads
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SAM ViT-H VELCRO — Value Locality Compression")
    p.add_argument("--mode", choices=AVAILABLE_MODES, required=True)

    p.add_argument("--checkpoint", type=str, help="Path to SAM ViT-H checkpoint (.pth)")
    p.add_argument("--coco_dir", type=str, help="Path to COCO dataset root")
    p.add_argument("--image_dir", type=str, default=None,
                    help="Path to flat image dirs (PTQ4ViT-style: <group>_profile/, <group>_eval/)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--domain_groups", type=str, default="general",
                    help="Comma-separated domain groups for profiling")
    p.add_argument("--profile_mlp", action="store_true",
                    help="Also profile MLP neurons (profile mode)")

    p.add_argument("--profile_dir", type=str, help="Profile results directory (freeze mode)")
    p.add_argument("--profile_group", type=str, help="Profile group name (freeze mode)")
    p.add_argument("--freeze_fraction", type=float, default=None)
    p.add_argument("--num_freeze_per_block", type=int, default=None)
    p.add_argument("--freeze_method", type=str,
                    choices=["rank", "ratio", "random", "entropy"], default="rank",
                    help="Freeze selection: rank (lowest variance), ratio (domain vs general), "
                         "random (seeded shuffle per block), entropy (lowest entropy)")
    p.add_argument("--seed", type=int, default=0,
                    help="Seed for stochastic ranking methods (random)")
    p.add_argument("--ratio_threshold", type=float, default=None,
                    help="For ratio method: freeze heads with variance reduction > threshold (0.0-1.0)")
    p.add_argument("--general_group", type=str, default="general",
                    help="General profile group name (for ratio method)")

    p.add_argument("--mlp_fraction", type=float, default=None,
                    help="Also freeze this fraction of MLP neurons (freeze mode: recorded in "
                         "manifest; eval mode: applied, overriding the manifest)")
    p.add_argument("--mlp_profile_dir", type=str, default=None,
                    help="MLP profile root (default: --profile_dir / manifest mlp config)")
    p.add_argument("--mlp_profile_group", type=str, default=None,
                    help="MLP profile group (default: --profile_group / manifest mlp config)")

    p.add_argument("--freeze_manifest", type=str, help="Freeze manifest JSON (eval mode)")
    p.add_argument("--eval_groups", type=str, default="general",
                    help="Comma-separated groups for evaluation")
    p.add_argument("--data_root", type=str, default="./data",
                    help="Root directory for GT masks (miss/false-positive decomposition)")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--save_visualizations", action="store_true")
    p.add_argument("--diagnostic_level", type=int, default=3,
                    help="0=masks only, 1=+activation, 2=+features, 3=+visual, 4=all")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "profile":
        if not args.checkpoint or (not args.coco_dir and not args.image_dir):
            print("ERROR: --checkpoint and (--coco_dir or --image_dir) required for profile mode")
            sys.exit(1)
        run_profile(args)

    elif args.mode == "freeze":
        if not args.profile_dir or not args.profile_group:
            print("ERROR: --profile_dir and --profile_group required for freeze mode")
            sys.exit(1)
        if args.freeze_method == "ratio":
            if args.ratio_threshold is None:
                print("ERROR: --ratio_threshold required for ratio freeze method")
                sys.exit(1)
        else:
            if args.freeze_fraction is None and args.num_freeze_per_block is None:
                print("ERROR: --freeze_fraction or --num_freeze_per_block required for rank method")
                sys.exit(1)
        run_freeze(args)

    elif args.mode == "eval":
        if not args.checkpoint or not args.freeze_manifest or (not args.coco_dir and not args.image_dir):
            print("ERROR: --checkpoint, --freeze_manifest, and (--coco_dir or --image_dir) required for eval mode")
            sys.exit(1)
        run_eval(args)


if __name__ == "__main__":
    main()
