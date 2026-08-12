"""Multi-level evaluation for SAM VELCRO experiments."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_boundary_iou(mask_a: np.ndarray, mask_b: np.ndarray, dilation: int = 1) -> float:
    kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
    boundary_a = cv2.dilate(mask_a.astype(np.uint8), kernel) - cv2.erode(mask_a.astype(np.uint8), kernel)
    boundary_b = cv2.dilate(mask_b.astype(np.uint8), kernel) - cv2.erode(mask_b.astype(np.uint8), kernel)
    intersection = np.logical_and(boundary_a, boundary_b).sum()
    union = np.logical_or(boundary_a, boundary_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _compute_iou_matrix_fast(masks_a: List[dict], masks_b: List[dict]) -> np.ndarray:
    """Compute pairwise IoU matrix using vectorized matrix multiplication.

    Downsamples masks to 256x256 for speed, then uses:
        intersection = A @ B.T
        union = area_a[:, None] + area_b[None, :] - intersection
        iou = intersection / union
    """
    n_a, n_b = len(masks_a), len(masks_b)

    # Get original shape from first mask
    h_orig, w_orig = masks_a[0]["segmentation"].shape
    # Downsample if large (>256 on either side)
    max_side = 256
    if h_orig > max_side or w_orig > max_side:
        scale = max_side / max(h_orig, w_orig)
        h_ds = max(1, int(h_orig * scale))
        w_ds = max(1, int(w_orig * scale))
        downsample = True
    else:
        h_ds, w_ds = h_orig, w_orig
        downsample = False

    n_pixels = h_ds * w_ds

    # Stack all masks into flat binary matrices
    mat_a = np.zeros((n_a, n_pixels), dtype=np.float32)
    mat_b = np.zeros((n_b, n_pixels), dtype=np.float32)

    for i in range(n_a):
        seg = masks_a[i]["segmentation"]
        if downsample:
            seg = cv2.resize(seg.astype(np.uint8), (w_ds, h_ds),
                             interpolation=cv2.INTER_NEAREST)
        mat_a[i] = seg.ravel().astype(np.float32)

    for j in range(n_b):
        seg = masks_b[j]["segmentation"]
        if downsample:
            seg = cv2.resize(seg.astype(np.uint8), (w_ds, h_ds),
                             interpolation=cv2.INTER_NEAREST)
        mat_b[j] = seg.ravel().astype(np.float32)

    # Vectorized IoU: intersection via dot product
    intersection = mat_a @ mat_b.T  # (n_a, n_b)
    area_a = mat_a.sum(axis=1)      # (n_a,)
    area_b = mat_b.sum(axis=1)      # (n_b,)
    union = area_a[:, None] + area_b[None, :] - intersection

    iou_matrix = np.where(union > 0, intersection / union, 0.0).astype(np.float32)
    return iou_matrix


def match_masks(
    masks_a: List[dict],
    masks_b: List[dict],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], set, set]:
    n_a, n_b = len(masks_a), len(masks_b)
    if n_a == 0 or n_b == 0:
        return [], set(range(n_a)), set(range(n_b))

    iou_matrix = _compute_iou_matrix_fast(masks_a, masks_b)

    row_idx, col_idx = linear_sum_assignment(-iou_matrix)

    matched = []
    matched_a, matched_b = set(), set()
    for i, j in zip(row_idx, col_idx):
        if iou_matrix[i, j] >= iou_threshold:
            matched.append((i, j, float(iou_matrix[i, j])))
            matched_a.add(i)
            matched_b.add(j)

    unmatched_a = set(range(n_a)) - matched_a
    unmatched_b = set(range(n_b)) - matched_b
    return matched, unmatched_a, unmatched_b


def evaluate_masks(
    masks_orig: List[dict],
    masks_patch: List[dict],
    iou_threshold: float = 0.5,
) -> dict:
    matched, unmatched_orig, unmatched_patch = match_masks(
        masks_orig, masks_patch, iou_threshold
    )

    ious = [m[2] for m in matched]
    n_orig = len(masks_orig)
    n_patch = len(masks_patch)

    return {
        "num_masks_original": n_orig,
        "num_masks_patched": n_patch,
        "mask_count_diff": n_patch - n_orig,
        "num_matched": len(matched),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "miss_rate": len(unmatched_orig) / n_orig if n_orig > 0 else 0.0,
        "false_pos_rate": len(unmatched_patch) / n_patch if n_patch > 0 else 0.0,
        "matched_pairs": matched,
    }


# ---------------------------------------------------------------------------
# Miss-rate classification (domain vs background)
# ---------------------------------------------------------------------------

DOMAIN_THRESHOLDS = {
    "kvasir":    {"iou": 0.3, "containment": 0.5},
    "isic":      {"iou": 0.3, "containment": 0.5},
    "busi":      {"iou": 0.3, "containment": 0.5},
    "deepglobe": {"iou": 0.1, "containment": 0.5},
    "pcb":       {"iou": 0.2, "containment": 0.3},
}


def is_domain_relevant(
    sam_mask: np.ndarray, gt_mask: np.ndarray, thresholds: dict
) -> Tuple[bool, float, float]:
    """Check if a single SAM mask is domain-relevant relative to a single GT mask.

    Returns (is_relevant, iou, containment).
    """
    intersection = np.logical_and(sam_mask, gt_mask).sum()
    union = np.logical_or(sam_mask, gt_mask).sum()
    sam_area = sam_mask.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    containment = float(intersection / sam_area) if sam_area > 0 else 0.0

    relevant = iou >= thresholds["iou"] or containment >= thresholds["containment"]
    return relevant, iou, containment


def gt_already_covered(
    gt_masks: List[np.ndarray],
    masks_orig: List[dict],
    matched_orig_indices: set,
    coverage_thresh: float = 0.7,
) -> Tuple[bool, float]:
    """Check if GT objects are already adequately covered by matched (non-missed) masks.

    Returns (is_covered, coverage_value).
    """
    if not gt_masks or not matched_orig_indices:
        return False, 0.0

    # Union all GT masks
    h, w = gt_masks[0].shape
    union_gt = np.zeros((h, w), dtype=bool)
    for gt in gt_masks:
        union_gt = np.logical_or(union_gt, gt)

    gt_area = union_gt.sum()
    if gt_area == 0:
        return True, 1.0

    # Union all matched original masks
    union_matched = np.zeros((h, w), dtype=bool)
    for idx in matched_orig_indices:
        seg = masks_orig[idx]["segmentation"]
        union_matched = np.logical_or(union_matched, seg)

    # Coverage = intersection(union_matched, union_gt) / area(union_gt)
    coverage = float(np.logical_and(union_matched, union_gt).sum() / gt_area)
    return coverage >= coverage_thresh, coverage


def _classify_unmatched(
    indices: set,
    masks: List[dict],
    gt_masks: List[np.ndarray],
    thresholds: dict,
    gt_covered: bool,
    has_gt: bool,
    idx_key: str,
) -> Tuple[int, int, int, List[dict]]:
    """Classify unmatched masks vs GT as domain / redundant / background.

    Shared by the miss (unmatched original) and false-positive (unmatched
    patched, i.e. added) decompositions.
    """
    domain_n = 0
    redundant_n = 0
    background_n = 0
    details = []

    for idx in sorted(indices):
        seg = masks[idx]["segmentation"]
        area = int(seg.sum())

        best_iou = 0.0
        best_containment = 0.0
        is_relevant = False

        if has_gt:
            for gt in gt_masks:
                relevant, iou_val, cont_val = is_domain_relevant(seg, gt, thresholds)
                if iou_val > best_iou:
                    best_iou = iou_val
                if cont_val > best_containment:
                    best_containment = cont_val
                if relevant:
                    is_relevant = True

        if is_relevant and not gt_covered:
            classification = "domain"
            domain_n += 1
        elif is_relevant and gt_covered:
            classification = "redundant"
            redundant_n += 1
        else:
            classification = "background"
            background_n += 1

        details.append({
            idx_key: idx,
            "area": area,
            "best_gt_iou": best_iou,
            "best_gt_containment": best_containment,
            "classification": classification,
        })

    return domain_n, redundant_n, background_n, details


def classify_misses(
    masks_orig: List[dict],
    masks_patch: List[dict],
    gt_masks: List[np.ndarray],
    domain: str,
    matched_pairs: Optional[List[Tuple[int, int, float]]] = None,
    iou_threshold: float = 0.5,
) -> dict:
    """Classify missed and added masks as domain-relevant, redundant, or background.

    Misses are unmatched original masks; added masks (false positives) are
    unmatched patched masks. Both sides are classified against GT with the
    same domain / redundant / background rules.

    Parameters
    ----------
    masks_orig : list of dict
        Original SAM masks (each with "segmentation" key).
    masks_patch : list of dict
        Patched SAM masks.
    gt_masks : list of np.ndarray
        Ground-truth binary masks for the image.
    domain : str
        Domain name (must be a key in DOMAIN_THRESHOLDS).
    matched_pairs : list of (orig_idx, patch_idx, iou), optional
        Pre-computed matched pairs. If None, will be computed.
    iou_threshold : float
        IoU threshold for mask matching.

    Returns
    -------
    dict with classification results for both misses and added masks.
    """
    has_gt = gt_masks is not None and len(gt_masks) > 0
    n_orig = len(masks_orig)
    n_patch = len(masks_patch)

    # Get thresholds for this domain
    thresholds = DOMAIN_THRESHOLDS.get(domain, {"iou": 0.3, "containment": 0.5})

    # Compute matching if not provided
    if matched_pairs is None:
        matched_pairs_list, unmatched_orig, unmatched_patch = match_masks(
            masks_orig, masks_patch, iou_threshold
        )
    else:
        matched_pairs_list = matched_pairs
        unmatched_orig = set(range(n_orig)) - {m[0] for m in matched_pairs_list}
        unmatched_patch = set(range(n_patch)) - {m[1] for m in matched_pairs_list}

    matched_orig_indices = {m[0] for m in matched_pairs_list}
    matched_patch_indices = {m[1] for m in matched_pairs_list}

    # Check GT coverage by matched masks (per side)
    if has_gt:
        gt_covered, gt_coverage = gt_already_covered(
            gt_masks, masks_orig, matched_orig_indices
        )
        gt_covered_patch, gt_coverage_patch = gt_already_covered(
            gt_masks, masks_patch, matched_patch_indices
        )
    else:
        gt_covered, gt_coverage = False, 0.0
        gt_covered_patch, gt_coverage_patch = False, 0.0

    # Classify each miss (unmatched original masks)
    domain_misses, redundant_misses, background_misses, per_miss_details = (
        _classify_unmatched(
            unmatched_orig, masks_orig, gt_masks, thresholds,
            gt_covered, has_gt, "orig_idx",
        )
    )

    # Classify each added mask (unmatched patched masks) the same way
    added_domain, added_redundant, added_background, per_added_details = (
        _classify_unmatched(
            unmatched_patch, masks_patch, gt_masks, thresholds,
            gt_covered_patch, has_gt, "patch_idx",
        )
    )

    total_misses = len(unmatched_orig)
    total_added = len(unmatched_patch)

    return {
        "total_misses": total_misses,
        "domain_misses": domain_misses,
        "redundant_misses": redundant_misses,
        "background_misses": background_misses,
        "domain_miss_rate": domain_misses / n_orig if n_orig > 0 else 0.0,
        "background_miss_rate": background_misses / n_orig if n_orig > 0 else 0.0,
        "gt_coverage": gt_coverage,
        "num_gt_masks": len(gt_masks) if gt_masks else 0,
        "has_gt": has_gt,
        "per_miss_details": per_miss_details,
        # False-positive (added mask) decomposition
        "total_added": total_added,
        "added_domain": added_domain,
        "added_redundant": added_redundant,
        "added_background": added_background,
        "added_domain_rate": added_domain / n_patch if n_patch > 0 else 0.0,
        "added_background_rate": added_background / n_patch if n_patch > 0 else 0.0,
        "gt_coverage_patched": gt_coverage_patch,
        "per_added_details": per_added_details,
    }


def compare_encoder_features(
    model,
    image: np.ndarray,
    device: torch.device,
    frozen_features: Optional[torch.Tensor] = None,
) -> dict:
    from segment_anything.utils.transforms import ResizeLongestSide

    transform = ResizeLongestSide(1024)
    input_image = transform.apply_image(image)
    input_tensor = torch.as_tensor(input_image, device=device).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor = model.preprocess(input_tensor)

    with torch.no_grad():
        features = model.image_encoder(input_tensor)  # (1, 256, 64, 64)

    if frozen_features is not None:
        cos_map = F.cosine_similarity(features, frozen_features, dim=1)  # (1, 64, 64)
        mean_sim = cos_map.mean().item()
        min_sim = cos_map.min().item()
        return {
            "mean_cosine_similarity": mean_sim,
            "min_cosine_similarity": min_sim,
            "cosine_map": cos_map.cpu().squeeze(0).numpy(),
            "features": features,
        }

    return {"features": features}


class MultiLevelEvaluator:
    """Orchestrates all 4 evaluation levels."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.mask_results: List[dict] = []
        self.feature_similarities: List[float] = []
        self.miss_classifications: List[dict] = []

    def add_mask_result(self, result: dict) -> None:
        self.mask_results.append(result)

    def add_feature_similarity(self, sim: float) -> None:
        self.feature_similarities.append(sim)

    def add_miss_classification(self, result: dict) -> None:
        self.miss_classifications.append(result)

    def aggregate(self) -> dict:
        if not self.mask_results:
            return {"error": "no results"}

        all_ious = []
        for r in self.mask_results:
            all_ious.extend([m[2] for m in r["matched_pairs"]])

        total_orig = sum(r["num_masks_original"] for r in self.mask_results)
        total_patch = sum(r["num_masks_patched"] for r in self.mask_results)
        total_matched = sum(r["num_matched"] for r in self.mask_results)

        agg = {
            "num_images": len(self.mask_results),
            "total_masks_original": total_orig,
            "total_masks_patched": total_patch,
            "total_matched": total_matched,
            "overall_mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "overall_median_iou": float(np.median(all_ious)) if all_ious else 0.0,
            "per_image_mean_iou": float(np.mean([r["mean_iou"] for r in self.mask_results])),
            "per_image_miss_rate": float(np.mean([r["miss_rate"] for r in self.mask_results])),
            "per_image_false_pos_rate": float(np.mean([r["false_pos_rate"] for r in self.mask_results])),
        }

        if self.feature_similarities:
            agg["mean_feature_cosine_sim"] = float(np.mean(self.feature_similarities))
            agg["min_feature_cosine_sim"] = float(np.min(self.feature_similarities))

        gt_results = [r for r in self.miss_classifications if r.get("has_gt")]
        if gt_results:
            agg["miss_decomposition"] = {
                "images_with_gt": len(gt_results),
                "total_misses": sum(r["total_misses"] for r in gt_results),
                "domain_misses": sum(r["domain_misses"] for r in gt_results),
                "redundant_misses": sum(r["redundant_misses"] for r in gt_results),
                "background_misses": sum(r["background_misses"] for r in gt_results),
            }
            agg["added_decomposition"] = {
                "images_with_gt": len(gt_results),
                "total_added": sum(r["total_added"] for r in gt_results),
                "added_domain": sum(r["added_domain"] for r in gt_results),
                "added_redundant": sum(r["added_redundant"] for r in gt_results),
                "added_background": sum(r["added_background"] for r in gt_results),
            }

        return agg
