"""COCO dataset loading with domain-group filtering for SAM."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def load_coco_annotations(coco_dir: Path) -> Tuple[dict, Dict[str, int]]:
    import json
    ann_path = coco_dir / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotations not found at {ann_path}")
    with open(ann_path) as f:
        coco = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    return coco, cat_name_to_id


def get_image_ids_for_domain(
    coco: dict,
    cat_name_to_id: Dict[str, int],
    category_names: Optional[List[str]],
) -> List[int]:
    if category_names is None:
        return [img["id"] for img in coco["images"]]
    target_cat_ids = {cat_name_to_id[n] for n in category_names if n in cat_name_to_id}
    if not target_cat_ids:
        return []
    image_ids = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids:
            image_ids.add(ann["image_id"])
    return sorted(image_ids)


def build_image_id_to_filename(coco: dict) -> Dict[int, str]:
    return {img["id"]: img["file_name"] for img in coco["images"]}


class COCODomainDataset:
    """Loads COCO val2017 images for a specific domain group."""

    def __init__(
        self,
        coco_dir: str,
        domain_name: str,
        category_names: Optional[List[str]],
        max_images: Optional[int] = None,
    ):
        self.coco_dir = Path(coco_dir)
        self.img_dir = self.coco_dir / "val2017"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"COCO val2017 images not found at {self.img_dir}")

        self.domain_name = domain_name
        self.coco, self.cat_name_to_id = load_coco_annotations(self.coco_dir)
        self.id_to_filename = build_image_id_to_filename(self.coco)

        image_ids = get_image_ids_for_domain(self.coco, self.cat_name_to_id, category_names)
        if max_images is not None and max_images < len(image_ids):
            image_ids = image_ids[:max_images]
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        img_id = self.image_ids[idx]
        filename = self.id_to_filename[img_id]
        img_path = self.img_dir / filename
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, img_id, filename

    def get_image_path(self, idx: int) -> Path:
        img_id = self.image_ids[idx]
        return self.img_dir / self.id_to_filename[img_id]


class ImageDirDataset:
    """Loads images from a directory tree (supports ILSVRC2012/PTQ4ViT layout)."""

    def __init__(
        self,
        image_dir: str,
        domain_name: str,
        max_images: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        if (self.image_dir / "val").is_dir():
            self.image_dir = self.image_dir / "val"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.domain_name = domain_name
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self.image_paths = sorted([
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ])

        if max_images is not None and max_images < len(self.image_paths):
            self.image_paths = self.image_paths[:max_images]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, idx, img_path.name

    def get_image_path(self, idx: int) -> Path:
        return self.image_paths[idx]


def create_domain_datasets(
    coco_dir: str,
    domain_groups: Dict[str, Optional[List[str]]],
    max_images: Optional[int] = None,
) -> Dict[str, COCODomainDataset]:
    datasets = {}
    for name, categories in domain_groups.items():
        ds = COCODomainDataset(coco_dir, name, categories, max_images)
        datasets[name] = ds
        print(f"  Domain '{name}': {len(ds)} images")
    return datasets
