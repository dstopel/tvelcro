#!/usr/bin/env python3
"""
generate_vit_comparison.py

Generate side-by-side classification comparisons: standard ViT vs mean-head ViT.
Shows each image with predicted class, confidence, and correctness for both networks.
Produces an optimization report with accuracy delta and timing.

Uses existing freeze_manifest.json — no re-profiling needed.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.models import get_net
import utils.datasets as datasets

# Try loading ImageNet class labels
try:
    import json as _json
    _IMAGENET_LABELS_PATH = None
    IMAGENET_LABELS = None
except Exception:
    IMAGENET_LABELS = None


# ==============================================================================
# Constants
# ==============================================================================

NUM_BLOCKS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN_DIM = 768
SEQ_LEN = 197


# ==============================================================================
# Utilities
# ==============================================================================

def get_imagenet_labels(model) -> Optional[Dict[int, str]]:
    """Try to get class index -> label mapping from timm model."""
    try:
        from timm.data import ImageNetInfo
        info = ImageNetInfo()
        return {i: info.index_to_label_name(i) for i in range(1000)}
    except Exception:
        pass
    # Fallback: try model's config
    try:
        cfg = model.default_cfg
        if "label_names" in cfg:
            return dict(enumerate(cfg["label_names"]))
    except Exception:
        pass
    return None


def load_font(size=16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except OSError:
            return ImageFont.load_default()


def confidence_interval(data: List[float], confidence: float = 0.95):
    n = len(data)
    if n < 2:
        m = np.mean(data) if data else 0
        return float(m), float(m), float(m)
    m = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n))
    if SCIPY_AVAILABLE:
        t_val = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        t_val = 1.96
    return m, m - t_val * se, m + t_val * se


# ==============================================================================
# Mean Head Patcher (imported from vit_value_locality_svelcro)
# ==============================================================================

from vit_value_locality_svelcro import ViTMeanHeadPatcher


# ==============================================================================
# Manifest Loading
# ==============================================================================

def load_manifest(path: str):
    with open(path) as f:
        manifest = json.load(f)
    frozen_heads = {}
    for e in manifest["frozen_heads"]:
        mean_path = e["mean_path"]
        if not Path(mean_path).exists():
            print(f"  [WARN] Missing: {mean_path}")
            continue
        frozen_heads[(e["block"], e["head"])] = np.load(mean_path)
    return frozen_heads, manifest


# ==============================================================================
# FLOP calculation
# ==============================================================================

def compute_flop_savings(frozen_entries):
    flops_per_head = 4 * SEQ_LEN * SEQ_LEN * HEAD_DIM + 5 * SEQ_LEN * SEQ_LEN
    total = NUM_BLOCKS * NUM_HEADS * flops_per_head

    frozen_per_block = defaultdict(set)
    for e in frozen_entries:
        frozen_per_block[e["block"]].add(e["head"])
    saved = sum(len(heads) for heads in frozen_per_block.values()) * flops_per_head

    return total, saved


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="ViT classification comparison: standard vs mean-head network"
    )
    ap.add_argument("--freeze_manifest", required=True)
    ap.add_argument("--image_dir", required=True,
                    help="ImageNet-style directory with val/ subfolder")
    ap.add_argument("--group_name", default=None,
                    help="Name for this group. Default: directory name.")
    ap.add_argument("--synset_words", default=None,
                    help="Path to synset_words.txt for correct ImageNet label mapping.")
    ap.add_argument("--model_name", default="vit_base_patch16_224")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Batch size=1 for per-image comparison")
    ap.add_argument("--num_images", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="comparison_vit")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    group_name = args.group_name or Path(args.image_dir).name

    # Load manifest
    print(f"Loading freeze manifest: {args.freeze_manifest}")
    frozen_heads, manifest = load_manifest(args.freeze_manifest)
    freeze_fraction = manifest.get("freeze_fraction", "?")
    num_freeze_per_block = manifest.get("num_freeze_per_block")
    profile_group = manifest.get("profile_group", "?")
    frozen_entries = manifest.get("frozen_heads", [])
    print(f"  Frozen heads: {len(frozen_heads)}")
    if num_freeze_per_block is not None:
        print(f"  T_k (per block): {num_freeze_per_block} ({profile_group})")
    else:
        print(f"  Freeze fraction: {freeze_fraction} ({profile_group})")

    total_flops, saved_flops = compute_flop_savings(frozen_entries)
    flop_pct = saved_flops / total_flops * 100 if total_flops > 0 else 0

    # Load model
    print("[INFO] Loading model...")
    net = get_net(args.model_name)
    device = next(net.parameters()).device

    # Try to get class labels
    labels = get_imagenet_labels(net)

    # Load data
    data_gen = datasets.ViTImageNetLoaderGenerator(
        args.image_dir, 'imagenet',
        args.batch_size, args.batch_size, args.num_workers,
        kwargs={"model": net}
    )

    # Apply ImageNet label mapping so target indices match pre-trained model output
    if args.synset_words:
        from vit_value_locality_svelcro import build_imagenet_label_mapping
        dataset = data_gen.test_set
        mapping = build_imagenet_label_mapping(args.synset_words, dataset.classes)
        dataset.target_transform = lambda y, m=mapping: m.get(y, y)
        print(f"  [label mapping] {len(dataset.classes)} classes mapped via {args.synset_words}")

    patcher = ViTMeanHeadPatcher(net, frozen_heads)

    # Warmup
    print("[INFO] Warmup...")
    with torch.inference_mode():
        loader = data_gen.test_loader()
        for inp, _ in loader:
            inp = inp.to(device)
            _ = net(inp)
            break
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── Phase A: Standard network ──
    print(f"\n[INFO] Phase A: Standard network ({args.num_images} images)...")
    std_results = []
    std_times = []

    with torch.inference_mode():
        loader = data_gen.test_loader()
        for i, (inp, target) in enumerate(loader):
            if i >= args.num_images:
                break
            inp = inp.to(device, non_blocking=True)
            target = target.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            output = net(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            std_times.append(elapsed)

            probs = F.softmax(output, dim=1)
            conf, pred = probs.max(1)
            correct = pred.eq(target).item()

            pred_idx = int(pred.item())
            pred_label = labels[pred_idx] if labels and pred_idx in labels else f"class_{pred_idx}"
            target_idx = int(target.item())
            target_label = labels[target_idx] if labels and target_idx in labels else f"class_{target_idx}"

            std_results.append({
                "index": i,
                "pred": pred_idx, "pred_label": pred_label,
                "target": target_idx, "target_label": target_label,
                "confidence": round(float(conf.item()), 4),
                "correct": correct,
                "time_sec": round(elapsed, 4),
            })

            mark = "OK" if correct else "WRONG"
            print(f"    [{i+1}] {mark} pred={pred_label} ({float(conf.item())*100:.1f}%) "
                  f"true={target_label} {elapsed:.3f}s")

    # ── Phase B: Mean-head network ──
    print(f"\n[INFO] Phase B: Mean-head network...")
    patcher.apply()

    # Warmup mean-head
    with torch.inference_mode():
        loader = data_gen.test_loader()
        for inp, _ in loader:
            inp = inp.to(device)
            _ = net(inp)
            break
    if device.type == "cuda":
        torch.cuda.synchronize()

    mh_results = []
    mh_times = []

    with torch.inference_mode():
        loader = data_gen.test_loader()
        for i, (inp, target) in enumerate(loader):
            if i >= args.num_images:
                break
            inp = inp.to(device, non_blocking=True)
            target = target.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            output = net(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            mh_times.append(elapsed)

            probs = F.softmax(output, dim=1)
            conf, pred = probs.max(1)
            correct = pred.eq(target).item()

            pred_idx = int(pred.item())
            pred_label = labels[pred_idx] if labels and pred_idx in labels else f"class_{pred_idx}"
            target_idx = int(target.item())
            target_label = labels[target_idx] if labels and target_idx in labels else f"class_{target_idx}"

            mh_results.append({
                "index": i,
                "pred": pred_idx, "pred_label": pred_label,
                "target": target_idx, "target_label": target_label,
                "confidence": round(float(conf.item()), 4),
                "correct": correct,
                "time_sec": round(elapsed, 4),
            })

            mark = "OK" if correct else "WRONG"
            print(f"    [{i+1}] {mark} pred={pred_label} ({float(conf.item())*100:.1f}%) "
                  f"true={target_label} {elapsed:.3f}s")

    patcher.restore()

    # ── Save comparison images ──
    print(f"\n[INFO] Saving comparison images...")
    font = load_font(14)
    font_sm = load_font(11)

    # Reload raw images for display
    from torchvision import transforms
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    raw_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    raw_dataset = datasets.ImageFolder(
        os.path.join(args.image_dir, 'val'), raw_transform
    )

    for i in range(min(args.num_images, len(std_results), len(mh_results), len(raw_dataset))):
        raw_img, _ = raw_dataset[i]
        if not isinstance(raw_img, Image.Image):
            raw_img = transforms.ToPILImage()(raw_img)

        s = std_results[i]
        m = mh_results[i]

        w, h = raw_img.size
        canvas = Image.new("RGB", (w * 2 + 20, h + 100), color=(255, 255, 255))

        # Left: standard
        canvas.paste(raw_img, (0, 50))
        # Right: same image (mean-head prediction differs, not the image itself)
        canvas.paste(raw_img, (w + 20, 50))

        draw = ImageDraw.Draw(canvas)

        # Standard label
        std_color = "green" if s["correct"] else "red"
        draw.text((5, 5), "Standard Network", fill="black", font=font)
        draw.text((5, 25), f"Pred: {s['pred_label']} ({s['confidence']*100:.1f}%)",
                  fill=std_color, font=font_sm)

        # Mean-head label
        mh_color = "green" if m["correct"] else "red"
        draw.text((w + 25, 5), "Mean-Head Network", fill="blue", font=font)
        draw.text((w + 25, 25), f"Pred: {m['pred_label']} ({m['confidence']*100:.1f}%)",
                  fill=mh_color, font=font_sm)

        # Ground truth + status
        draw.text((5, h + 55), f"True: {s['target_label']}", fill="gray", font=font_sm)

        status = ""
        if s["correct"] and m["correct"]:
            status = "Both correct"
            status_color = "green"
        elif s["correct"] and not m["correct"]:
            status = "Mean-head WRONG (regression)"
            status_color = "red"
        elif not s["correct"] and m["correct"]:
            status = "Mean-head FIXED (improvement)"
            status_color = "blue"
        else:
            status = "Both wrong"
            status_color = "orange"

        draw.text((5, h + 72), status, fill=status_color, font=font)
        draw.text((5, h + 87), f"Std: {s['time_sec']:.3f}s  MeanHead: {m['time_sec']:.3f}s",
                  fill="gray", font=font_sm)

        canvas.save(out_dir / f"compare_{i:03d}.png")

    # ── Statistics ──
    std_correct = sum(1 for r in std_results if r["correct"])
    mh_correct = sum(1 for r in mh_results if r["correct"])
    n = len(std_results)
    std_acc = std_correct / n * 100 if n > 0 else 0
    mh_acc = mh_correct / n * 100 if n > 0 else 0

    speedups = [s / m if m > 0 else 0 for s, m in zip(std_times, mh_times)]
    mean_speedup, ci_lo, ci_hi = confidence_interval(speedups)
    mean_std_time, _, _ = confidence_interval(std_times)
    mean_mh_time, _, _ = confidence_interval(mh_times)

    # Count regressions and improvements
    regressions = sum(1 for s, m in zip(std_results, mh_results)
                      if s["correct"] and not m["correct"])
    improvements = sum(1 for s, m in zip(std_results, mh_results)
                       if not s["correct"] and m["correct"])

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("ViT CLASSIFICATION COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Freeze manifest   : {args.freeze_manifest}")
    lines.append(f"Model             : {args.model_name}")
    if num_freeze_per_block is not None:
        lines.append(f"T_k (per block)   : {num_freeze_per_block}")
    else:
        lines.append(f"Freeze fraction   : {freeze_fraction}")
    lines.append(f"Profile group     : {profile_group}")
    lines.append(f"Eval group        : {group_name}")
    lines.append(f"Images evaluated  : {n}")
    lines.append("")

    # Frozen heads
    lines.append("-" * 80)
    lines.append("FROZEN HEADS")
    lines.append("-" * 80)
    lines.append(f"{'Block':<8} {'Head':<6} {'Channel Rank':<18}")
    lines.append("-" * 35)
    for e in sorted(frozen_entries, key=lambda x: (x["block"], x["head"])):
        cr = e.get("channel_rank", 0)
        lines.append(f"{e['block']:<8} {e['head']:<6} {cr:<18.6f}")

    # Per-block summary
    lines.append("")
    frozen_per_block = defaultdict(set)
    for e in frozen_entries:
        frozen_per_block[e["block"]].add(e["head"])

    lines.append(f"{'Block':<8} {'Active':<20} {'Mean Heads':<20}")
    lines.append("-" * 50)
    for b in range(NUM_BLOCKS):
        fh = frozen_per_block.get(b, set())
        active = sorted(set(range(NUM_HEADS)) - fh)
        frozen = sorted(fh)
        a_str = ",".join(str(h) for h in active) if active else "-"
        f_str = ",".join(str(h) for h in frozen) if frozen else "-"
        lines.append(f"{b:<8} {a_str:<20} {f_str:<20}")

    # Accuracy
    lines.append("")
    lines.append("-" * 80)
    lines.append("ACCURACY COMPARISON")
    lines.append("-" * 80)
    lines.append(f"Standard network  : {std_acc:.1f}% ({std_correct}/{n})")
    lines.append(f"Mean-head network : {mh_acc:.1f}% ({mh_correct}/{n})")
    lines.append(f"Delta             : {mh_acc - std_acc:+.1f}%")
    lines.append(f"Regressions       : {regressions} (standard correct -> mean-head wrong)")
    lines.append(f"Improvements      : {improvements} (standard wrong -> mean-head correct)")

    # FLOP savings
    lines.append("")
    lines.append("-" * 80)
    lines.append("THEORETICAL ATTENTION FLOP SAVINGS")
    lines.append("-" * 80)
    lines.append(f"Total attention FLOPs   : {total_flops:>15,}")
    lines.append(f"Saved attention FLOPs   : {saved_flops:>15,}")
    lines.append(f"Reduction               : {flop_pct:.2f}%")

    # Timing
    lines.append("")
    lines.append("-" * 80)
    lines.append("TIMING (95% confidence interval)")
    lines.append("-" * 80)
    lines.append(f"Mean standard time  : {mean_std_time:.4f} sec")
    lines.append(f"Mean mean-head time : {mean_mh_time:.4f} sec")
    lines.append(f"Mean speedup        : {mean_speedup:.4f}x  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    time_saved = (1 - mean_mh_time / mean_std_time) * 100 if mean_std_time > 0 else 0
    lines.append(f"Wall-clock saved    : {time_saved:.2f}%")

    # Per-image table
    lines.append("")
    lines.append("-" * 80)
    lines.append("PER-IMAGE RESULTS")
    lines.append("-" * 80)
    lines.append(f"{'#':<4} {'Std Pred':<18} {'MH Pred':<18} {'True':<18} {'Std':<5} {'MH':<5} {'Delta':<8}")
    lines.append("-" * 80)
    for s, m in zip(std_results, mh_results):
        s_mark = "OK" if s["correct"] else "X"
        m_mark = "OK" if m["correct"] else "X"
        delta = ""
        if s["correct"] and not m["correct"]:
            delta = "REGR"
        elif not s["correct"] and m["correct"]:
            delta = "IMPR"
        lines.append(f"{s['index']:<4} {s['pred_label'][:17]:<18} {m['pred_label'][:17]:<18} "
                     f"{s['target_label'][:17]:<18} {s_mark:<5} {m_mark:<5} {delta:<8}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("METHOD: Mean Head Replacement")
    lines.append("- Frozen heads' Q/K/V zeroed in fused self.qkv weight")
    lines.append("- Pre-hook on self.proj overwrites frozen slots with precomputed mean")
    lines.append("- Network architecture unchanged (12 heads per block)")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n{report_text}")
    print(f"\nReport: {report_path}")

    # Save JSON
    meta = {
        "version": "1.5-vit-svelcro",
        "freeze_manifest": args.freeze_manifest,
        "model": args.model_name,
        "freeze_fraction": freeze_fraction,
        "num_freeze_per_block": num_freeze_per_block,
        "profile_group": profile_group,
        "eval_group": group_name,
        "n_images": n,
        "accuracy": {
            "standard_top1": round(std_acc, 2),
            "mean_head_top1": round(mh_acc, 2),
            "delta": round(mh_acc - std_acc, 2),
            "regressions": regressions,
            "improvements": improvements,
        },
        "timing": {
            "mean_standard_sec": round(mean_std_time, 4),
            "mean_mean_head_sec": round(mean_mh_time, 4),
            "mean_speedup": round(mean_speedup, 4),
            "speedup_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "flop_savings_pct": round(flop_pct, 2),
        "standard_results": std_results,
        "mean_head_results": mh_results,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {n} comparisons saved to {out_dir}/")


if __name__ == "__main__":
    main()

