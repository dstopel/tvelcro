#!/usr/bin/env python3
"""Show that T-VELCRO composes with post-training quantization.

The claim under test: value-locality removal and quantization target different
redundancies (which units run, versus how precisely each remaining weight is
stored), so the savings should stack rather than overlap.

Four variants are compared at a fixed operating point:
    fp        full precision, all heads
    fp+tv     full precision, T-VELCRO structural head removal
    int8      quantized (dynamic int8 on Linear layers), all heads
    int8+tv   T-VELCRO removal, then quantization

Order matters: units are removed FIRST, then what remains is quantized. Slicing
an already-quantized tensor is fragile, and remove-then-quantize is also the
realistic deployment order.

Reported per variant: top-1 on the eval set, parameter-storage bytes of the
encoder Linear layers, and CPU latency (dynamic int8 kernels are CPU-side, so
the timing comparison is run on CPU for all variants to stay apples-to-apples).

Usage (from repo root, PYTHONPATH=.):
  python experiments/vit/bench_quant_compose.py \
      --model vit_base_patch16_224 \
      --manifest results/vit_sweep/vit_base/seed0/tk3/freeze_dogs/freeze_manifest.json \
      --eval_dir data/vit_groups/dogs_eval --synset_words data/synset_words.txt \
      --out results/quant_compose/vit_base_dogs.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vit"))


def linear_storage_bytes(model):
    """Bytes held by Linear weights, counting quantized modules at int8."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel() * m.weight.element_size()
            if m.bias is not None:
                total += m.bias.numel() * m.bias.element_size()
        elif m.__class__.__name__ in ("DynamicQuantizedLinear", "LinearPackedParams"):
            try:
                w, b = m._weight_bias()
                total += w.numel() * 1  # int8
                if b is not None:
                    total += b.numel() * b.element_size()
            except Exception:
                pass
    return total


@torch.no_grad()
def evaluate(model, loader, device, label_map, max_batches=None):
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits = model(x.to(device))
        if hasattr(logits, "logits"):
            logits = logits.logits
        pred = logits.argmax(-1).cpu()
        mapped = torch.tensor([label_map.get(int(t), -1) for t in y])
        correct += (pred == mapped).sum().item()
        total += len(y)
    return 100.0 * correct / max(1, total), total


@torch.no_grad()
def bench_cpu(model, batch=1, size=224, warmup=3, iters=10):
    x = torch.randn(batch, 3, size, size)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--manifest", required=True)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--synset_words", default="data/synset_words.txt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import timm
    from torch.ao.quantization import quantize_dynamic
    from torchvision import datasets, transforms
    from surgery import build_sliced_vit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fresh():
        m = timm.create_model(args.model, pretrained=True)
        m.eval()
        return m

    # dataset + label mapping (folder name -> imagenet index)
    cfg = timm.data.resolve_data_config({}, model=fresh())
    tf = timm.data.create_transform(**cfg)
    root = Path(args.eval_dir)
    if (root / "val").is_dir():
        root = root / "val"
    ds = datasets.ImageFolder(str(root), transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=4)

    wnids = [l.split()[0] for l in open(args.synset_words)]
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    label_map = {i: wnid_to_idx.get(c, -1) for c, i in ds.class_to_idx.items()}
    label_map = {v: wnid_to_idx.get(k, -1) for k, v in ds.class_to_idx.items()}

    with open(args.manifest) as f:
        manifest = json.load(f)
    n_frozen = len(manifest["frozen_heads"])

    results = {}

    # fp, all heads
    m = fresh().to(device)
    acc, n = evaluate(m, loader, device, label_map, args.max_batches)
    results["fp"] = {"top1": acc, "n": n,
                     "linear_bytes": linear_storage_bytes(m),
                     "cpu_ms_b1": bench_cpu(fresh())}

    # fp + T-VELCRO
    m = fresh()
    build_sliced_vit(m, manifest)
    m_dev = m.to(device)
    acc, n = evaluate(m_dev, loader, device, label_map, args.max_batches)
    m_cpu = fresh(); build_sliced_vit(m_cpu, manifest)
    results["fp+tvelcro"] = {"top1": acc, "n": n,
                             "linear_bytes": linear_storage_bytes(m_cpu),
                             "cpu_ms_b1": bench_cpu(m_cpu)}

    # int8, all heads
    mq = quantize_dynamic(fresh(), {nn.Linear}, dtype=torch.qint8)
    acc, n = evaluate(mq, loader, torch.device("cpu"), label_map, args.max_batches)
    results["int8"] = {"top1": acc, "n": n,
                       "linear_bytes": linear_storage_bytes(mq),
                       "cpu_ms_b1": bench_cpu(mq)}

    # T-VELCRO first, then int8: the composition under test
    m = fresh()
    build_sliced_vit(m, manifest)
    mq = quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    acc, n = evaluate(mq, loader, torch.device("cpu"), label_map, args.max_batches)
    results["int8+tvelcro"] = {"top1": acc, "n": n,
                               "linear_bytes": linear_storage_bytes(mq),
                               "cpu_ms_b1": bench_cpu(mq)}

    base_b = results["fp"]["linear_bytes"]
    base_ms = results["fp"]["cpu_ms_b1"]
    for k, v in results.items():
        v["storage_vs_fp"] = round(v["linear_bytes"] / base_b, 4)
        v["cpu_speedup_vs_fp"] = round(base_ms / v["cpu_ms_b1"], 3)

    out = {"config": {"model": args.model, "manifest": args.manifest,
                      "frozen_heads": n_frozen, "eval_dir": args.eval_dir,
                      "images": results["fp"]["n"]},
           "variants": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'variant':<14} {'top1':>7} {'storage':>9} {'cpu ms':>8} {'speedup':>8}")
    print("-" * 50)
    for k, v in results.items():
        print(f"{k:<14} {v['top1']:>6.2f}% {v['storage_vs_fp']:>8.3f}x "
              f"{v['cpu_ms_b1']:>7.1f} {v['cpu_speedup_vs_fp']:>7.2f}x")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
