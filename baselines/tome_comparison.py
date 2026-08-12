#!/usr/bin/env python3
"""ToMe token-merging baseline for the T-VELCRO ViT comparison.

Comparison protocol (matched analytical FLOPs):
  ToMe reduces the token count S by merging r tokens per transformer block;
  T-VELCRO keeps S constant and reduces the number of computed units
  (frozen attention heads / MLP slices). The two are compared at operating
  points where the analytical MAC reduction over the 12 transformer blocks
  matches the T-VELCRO operating points (25% and 41.7%). Patch embedding
  and the classifier head are excluded from the count on both sides.

  Accuracy: top-1 on the same per-domain eval dirs produced by
  vit/create_groups.py (ImageFolder layout, folder names mapped to
  ImageNet-1k indices via synset_words.txt).
  Latency: common.timing.measure_latency on a fixed random batch.

Run from repo root:
  PYTHONPATH=. python baselines/tome_comparison.py \\
      --data_dir ./data/vit_groups --domains snakes fish boats birds
"""

import argparse
import json
from pathlib import Path

import timm
import tome
import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from common.timing import measure_latency

# vit_base_patch16_224 geometry
S0, D, LAYERS = 197, 768, 12


def _block_macs(s_attn: int, s_mlp: int) -> int:
    attn = 4 * s_attn * D * D + 2 * s_attn * s_attn * D  # qkv+proj, scores+values
    mlp = 8 * s_mlp * D * D
    return attn + mlp


def flop_reduction(r: int) -> float:
    """Analytical MAC reduction over the 12 blocks for constant per-block r.

    ToMe merges r tokens between attention and MLP in every block, so layer i
    attends over S0 - i*r tokens and its MLP sees S0 - (i+1)*r.
    """
    base = LAYERS * _block_macs(S0, S0)
    s, total = S0, 0
    for _ in range(LAYERS):
        s_next = max(s - r, 2)  # the cls token is never merged away
        total += _block_macs(s, s_next)
        s = s_next
    return 1.0 - total / base


def match_r(target: float) -> int:
    """Smallest-error r for a target fractional FLOP reduction."""
    return min(range(41), key=lambda r: abs(flop_reduction(r) - target))


def build_label_mapping(synset_words_path: str, class_names) -> dict:
    """ImageFolder local index -> ImageNet-1k index.

    Accepts wnid folders (n01440764), joined-name folders as produced by
    create_groups (tench_Tinca_tinca), or first-name folders (tench).
    """
    lut = {}
    with open(synset_words_path) as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            lut[parts[0]] = idx  # wnid
            synonyms = [s.strip() for s in parts[1].split(",")]
            joined = "_".join(s.replace(" ", "_") for s in synonyms)
            first = synonyms[0].replace(" ", "_")
            for key in (joined, joined.lower(), first, first.lower()):
                lut[key] = idx
    mapping = {}
    for local_idx, name in enumerate(class_names):
        target = lut.get(name, lut.get(name.lower()))
        if target is None:
            raise KeyError(f"class folder {name!r} not found in {synset_words_path}")
        mapping[local_idx] = target
    return mapping


def resolve_eval_dir(data_dir: Path, domain: str) -> Path:
    for cand in (data_dir / f"{domain}_eval" / "val", data_dir / f"{domain}_eval",
                 data_dir / domain / "eval", data_dir / domain):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"no eval dir for domain {domain!r} under {data_dir}")


@torch.inference_mode()
def eval_top1(model, dataset, mapping, batch_size, device) -> float:
    remap = torch.tensor([mapping[i] for i in range(len(mapping))], device=device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == remap[y]).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", required=True, help="dir holding <domain>_eval/val trees from vit/create_groups.py")
    p.add_argument("--domains", nargs="+",
                   default=["dogs", "cats", "cars", "snakes", "fish", "boats", "birds"])
    p.add_argument("--r_values", nargs="+", type=int, default=None,
                   help="ToMe r per block; default: 0 plus r matched to 25%% and 41.7%% FLOP reduction")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_images", type=int, default=300, help="cap eval images per domain")
    p.add_argument("--out", default="baselines/results/tome_comparison.json")
    p.add_argument("--synset_words", default="./data/synset_words.txt",
                   help="ImageNet synset_words.txt for folder-name to class-index mapping")
    args = p.parse_args()

    assert flop_reduction(0) == 0.0  # sanity check on the analytical model
    if args.r_values is None:
        args.r_values = [0] + [match_r(t) for t in (0.25, 0.417)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    results = {}

    for r in args.r_values:
        model = timm.create_model("vit_base_patch16_224", pretrained=True).eval().to(device)
        transform = create_transform(**resolve_data_config({}, model=model))
        tome.patch.timm(model)
        model.r = r
        red = flop_reduction(r)
        print(f"[r={r}] analytical block-FLOP reduction: {red * 100:.1f}%")

        x = torch.randn(args.batch_size, 3, 224, 224, device=device)
        lat = measure_latency(lambda: model(x))
        if not isinstance(lat, dict):
            lat = {"latency": lat}
        print(f"[r={r}] latency (batch={args.batch_size}): {lat}")

        entry = {"analytical_flop_reduction": red, "latency": lat, "domains": {}}
        for domain in args.domains:
            dataset = ImageFolder(resolve_eval_dir(data_dir, domain), transform=transform)
            mapping = build_label_mapping(args.synset_words, dataset.classes)
            if args.max_images and len(dataset) > args.max_images:
                dataset = Subset(dataset, range(args.max_images))
            top1 = eval_top1(model, dataset, mapping, args.batch_size, device)
            n = len(dataset)
            entry["domains"][domain] = {"top1": top1, "n": n}
            print(f"[r={r}] {domain}: top1={top1 * 100:.2f}% (n={n})")
        results[str(r)] = entry
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
