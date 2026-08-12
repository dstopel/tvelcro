"""Utility functions for SAM VELCRO experiments."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch


def setup_logging(out_dir: Path, name: str = "sam_velcro") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {name} ({mem_gb:.1f} GB)")
        return dev
    print("Using CPU")
    return torch.device("cpu")


def load_sam_model(checkpoint_path: str, device: torch.device):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return sam


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            print(f"[{self.name}] {self.elapsed:.2f}s")
