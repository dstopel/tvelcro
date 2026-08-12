"""Ground-truth mask loading for cross-domain miss rate analysis.

Each of the 5 cross-domain datasets (Kvasir-SEG, ISIC 2018, BUSI, DeepGlobe, PCB)
has different GT mask formats and filename conventions. This module provides a
unified interface for loading binary GT masks.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kvasir-SEG
# ---------------------------------------------------------------------------

def _find_kvasir_gt_path(image_filename: str, data_root: str) -> Optional[Path]:
    """Resolve GT mask path for Kvasir-SEG.

    GT location: {data_root}/kvasir-seg/masks/{filename}
    Same filename as the image (both .jpg or .png).
    """
    masks_dir = Path(data_root) / "kvasir-seg" / "masks"
    gt_path = masks_dir / image_filename
    if gt_path.exists():
        return gt_path
    # Try alternate extensions
    stem = Path(image_filename).stem
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_kvasir_gt(image_filename: str, data_root: str) -> List[np.ndarray]:
    """Load Kvasir-SEG GT mask (binary polyp segmentation)."""
    gt_path = _find_kvasir_gt_path(image_filename, data_root)
    if gt_path is None:
        logger.warning("Kvasir GT not found for: %s", image_filename)
        return []

    mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Failed to read Kvasir GT: %s", gt_path)
        return []

    binary = (mask > 127).astype(np.uint8)
    return [binary]


# ---------------------------------------------------------------------------
# ISIC 2018
# ---------------------------------------------------------------------------

def _find_isic_gt_path(image_filename: str, data_root: str) -> Optional[Path]:
    """Resolve GT mask path for ISIC 2018.

    Image: ISIC_0024306.jpg -> GT: ISIC_0024306_segmentation.png
    """
    stem = Path(image_filename).stem
    gt_dir = (
        Path(data_root) / "ISIC2018" / "ISIC2018_Task1_Training_GroundTruth"
    )
    gt_path = gt_dir / f"{stem}_segmentation.png"
    if gt_path.exists():
        return gt_path
    # Fallback: try other extensions or case-insensitive match
    for ext in (".png", ".jpg", ".bmp"):
        candidate = gt_dir / f"{stem}_segmentation{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_isic_gt(image_filename: str, data_root: str) -> List[np.ndarray]:
    """Load ISIC 2018 GT mask (binary lesion segmentation)."""
    gt_path = _find_isic_gt_path(image_filename, data_root)
    if gt_path is None:
        logger.warning("ISIC GT not found for: %s", image_filename)
        return []

    mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Failed to read ISIC GT: %s", gt_path)
        return []

    binary = (mask > 127).astype(np.uint8)
    return [binary]


# ---------------------------------------------------------------------------
# BUSI (Breast Ultrasound Images)
# ---------------------------------------------------------------------------

def _unsanitize_busi_filename(sanitized: str) -> str:
    """Reverse the sanitization applied to BUSI filenames.

    Sanitization rule: tr ' ()' '_[]'
      - space -> underscore
      - '(' -> '['
      - ')' -> ']'

    Reverse:
      - '_[' -> ' ('   (underscore before bracket was originally space before paren)
      - ']' -> ')'
    """
    # Replace '_[' with ' (' and ']' with ')'
    result = sanitized.replace("_[", " (").replace("]", ")")
    return result


def _sanitize_for_comparison(name: str) -> str:
    """Apply sanitization for comparison purposes."""
    return name.replace(" ", "_").replace("(", "[").replace(")", "]")


def _find_busi_gt_paths(image_filename: str, data_root: str) -> List[Path]:
    """Find GT mask file(s) for BUSI dataset.

    BUSI masks are alongside images: {name}_mask.png, {name}_mask_1.png, etc.
    Normal class has no GT mask.
    """
    busi_dir = Path(data_root) / "busi"
    stem = Path(image_filename).stem

    # Attempt to reverse the sanitization to get original filename stem
    original_stem = Path(_unsanitize_busi_filename(image_filename)).stem

    # Search in all 3 subdirectories
    subdirs = ["benign", "malignant", "normal"]
    found_masks: List[Path] = []

    for subdir in subdirs:
        subdir_path = busi_dir / subdir
        if not subdir_path.exists():
            continue

        # Strategy 1: Try exact original name match
        mask_path = subdir_path / f"{original_stem}_mask.png"
        if mask_path.exists():
            found_masks.append(mask_path)
            # Check for additional masks (_mask_1.png, _mask_2.png, ...)
            for i in range(1, 10):
                extra = subdir_path / f"{original_stem}_mask_{i}.png"
                if extra.exists():
                    found_masks.append(extra)
                else:
                    break
            return found_masks

        # Strategy 2: Try sanitized stem directly (in case data is already sanitized)
        mask_path = subdir_path / f"{stem}_mask.png"
        if mask_path.exists():
            found_masks.append(mask_path)
            for i in range(1, 10):
                extra = subdir_path / f"{stem}_mask_{i}.png"
                if extra.exists():
                    found_masks.append(extra)
                else:
                    break
            return found_masks

    # Strategy 3: Fallback - glob all mask files and fuzzy match
    for subdir in subdirs:
        subdir_path = busi_dir / subdir
        if not subdir_path.exists():
            continue

        for mask_file in subdir_path.glob("*_mask*.png"):
            # Extract the base name (everything before _mask)
            mask_name = mask_file.name
            # Find the last occurrence of _mask to split
            mask_idx = mask_name.find("_mask")
            if mask_idx == -1:
                continue
            base_name = mask_name[:mask_idx]
            # Compare sanitized versions
            if _sanitize_for_comparison(base_name) == stem:
                found_masks.append(mask_file)

    # Deduplicate and sort
    found_masks = sorted(set(found_masks))
    return found_masks


def _load_busi_gt(image_filename: str, data_root: str) -> List[np.ndarray]:
    """Load BUSI GT mask(s).

    Normal class images have no GT mask -> returns empty list.
    Some lesion images have multiple masks (multi-focal).
    """
    gt_paths = _find_busi_gt_paths(image_filename, data_root)
    if not gt_paths:
        # This may be a normal-class image (expected) or a missing file (warning)
        stem = Path(image_filename).stem
        original = _unsanitize_busi_filename(image_filename)
        if "normal" in original.lower() or "normal" in stem.lower():
            logger.debug("BUSI normal class (no GT expected): %s", image_filename)
        else:
            logger.warning("BUSI GT not found for: %s", image_filename)
        return []

    masks = []
    for gt_path in gt_paths:
        mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Failed to read BUSI GT: %s", gt_path)
            continue
        binary = (mask > 127).astype(np.uint8)
        # Skip empty masks (all zeros)
        if binary.sum() > 0:
            masks.append(binary)

    return masks


# ---------------------------------------------------------------------------
# DeepGlobe (Road Extraction)
# ---------------------------------------------------------------------------

def _find_deepglobe_gt_path(image_filename: str, data_root: str) -> Optional[Path]:
    """Resolve GT mask path for DeepGlobe.

    Image: {id}_sat.png -> GT: {id}_mask.png
    Search in train/, valid/, test/ subdirectories.
    """
    stem = Path(image_filename).stem
    # Replace _sat suffix with _mask
    if stem.endswith("_sat"):
        mask_stem = stem[:-4] + "_mask"
    else:
        mask_stem = stem + "_mask"

    deepglobe_dir = Path(data_root) / "deepglobe"

    # Search in common subdirectories
    search_dirs = [
        deepglobe_dir / "train",
        deepglobe_dir / "valid",
        deepglobe_dir / "test",
        deepglobe_dir,  # root level fallback
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in (".png", ".tif", ".jpg", ".jpeg"):
            candidate = search_dir / f"{mask_stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def _load_deepglobe_gt(image_filename: str, data_root: str) -> List[np.ndarray]:
    """Load DeepGlobe GT mask (road network).

    GT is an RGB image where white pixels (>200 in all channels) indicate road.
    """
    gt_path = _find_deepglobe_gt_path(image_filename, data_root)
    if gt_path is None:
        logger.warning("DeepGlobe GT not found for: %s", image_filename)
        return []

    mask_rgb = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    if mask_rgb is None:
        logger.warning("Failed to read DeepGlobe GT: %s", gt_path)
        return []

    # White pixels (>200 in all BGR channels) = road
    white_mask = (
        (mask_rgb[:, :, 0] > 200)
        & (mask_rgb[:, :, 1] > 200)
        & (mask_rgb[:, :, 2] > 200)
    )
    binary = white_mask.astype(np.uint8)
    return [binary]


# ---------------------------------------------------------------------------
# PCB (Defect Detection with VOC XML annotations)
# ---------------------------------------------------------------------------

def _find_pcb_xml_path(image_filename: str, data_root: str) -> Optional[Path]:
    """Find PASCAL VOC XML annotation for a PCB image.

    Searches recursively in {data_root}/pcb/ for {stem}.xml
    """
    stem = Path(image_filename).stem
    pcb_dir = Path(data_root) / "pcb"

    if not pcb_dir.exists():
        return None

    # Direct path attempts first (common locations)
    common_dirs = [
        pcb_dir / "Annotations",
        pcb_dir / "annotations",
        pcb_dir,
    ]
    for d in common_dirs:
        if d.exists():
            candidate = d / f"{stem}.xml"
            if candidate.exists():
                return candidate

    # Recursive fallback
    matches = list(pcb_dir.rglob(f"{stem}.xml"))
    if matches:
        return matches[0]

    return None


def _load_pcb_gt(image_filename: str, data_root: str) -> List[np.ndarray]:
    """Load PCB GT from PASCAL VOC XML (bounding box annotations).

    Each <object> bounding box becomes a filled rectangle binary mask.
    """
    xml_path = _find_pcb_xml_path(image_filename, data_root)
    if xml_path is None:
        logger.warning("PCB XML not found for: %s", image_filename)
        return []

    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        logger.warning("Failed to parse PCB XML %s: %s", xml_path, e)
        return []

    root = tree.getroot()

    # Get image dimensions from <size> element
    size_elem = root.find("size")
    if size_elem is None:
        logger.warning("No <size> element in PCB XML: %s", xml_path)
        return []

    width_elem = size_elem.find("width")
    height_elem = size_elem.find("height")
    if width_elem is None or height_elem is None:
        logger.warning("Missing width/height in PCB XML: %s", xml_path)
        return []

    try:
        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
    except (ValueError, TypeError):
        logger.warning("Invalid width/height values in PCB XML: %s", xml_path)
        return []

    if img_width <= 0 or img_height <= 0:
        logger.warning("Non-positive dimensions in PCB XML: %s", xml_path)
        return []

    # Parse all <object> bounding boxes
    masks = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
        except (ValueError, TypeError, AttributeError):
            logger.warning("Invalid bndbox in object, skipping: %s", xml_path)
            continue

        # Clamp to image bounds
        xmin = max(0, min(xmin, img_width - 1))
        ymin = max(0, min(ymin, img_height - 1))
        xmax = max(0, min(xmax, img_width))
        ymax = max(0, min(ymax, img_height))

        if xmax <= xmin or ymax <= ymin:
            continue

        # Create filled rectangle mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax] = 1
        masks.append(mask)

    if not masks:
        logger.warning("No valid bounding boxes in PCB XML: %s", xml_path)

    return masks


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_DOMAIN_LOADERS = {
    "kvasir": _load_kvasir_gt,
    "isic": _load_isic_gt,
    "busi": _load_busi_gt,
    "deepglobe": _load_deepglobe_gt,
    "pcb": _load_pcb_gt,
}


def load_gt_masks(
    image_filename: str,
    domain: str,
    data_root: str = "./data",
) -> List[np.ndarray]:
    """Load binary GT masks for the given image.

    Args:
        image_filename: The filename of the eval image (may be sanitized
            with tr ' ()' '_[]').
        domain: One of "kvasir", "isic", "busi", "deepglobe", "pcb".
        data_root: Root data directory (default ./data).

    Returns:
        List of binary numpy arrays (H, W), dtype=uint8, values 0 or 1.
        Empty list if no GT exists (e.g. BUSI normal class).
    """
    domain_lower = domain.lower().strip()

    loader = _DOMAIN_LOADERS.get(domain_lower)
    if loader is None:
        logger.error(
            "Unknown domain '%s'. Supported: %s",
            domain,
            list(_DOMAIN_LOADERS.keys()),
        )
        return []

    try:
        masks = loader(image_filename, data_root)
    except Exception as e:
        logger.error(
            "Unexpected error loading GT for %s (domain=%s): %s",
            image_filename,
            domain,
            e,
        )
        return []

    # Validate outputs
    validated = []
    for i, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            logger.warning("GT mask %d is not ndarray for %s", i, image_filename)
            continue
        if mask.ndim != 2:
            logger.warning("GT mask %d is not 2D for %s (shape=%s)", i, image_filename, mask.shape)
            continue
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        # Ensure strictly binary
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        validated.append(mask)

    return validated
