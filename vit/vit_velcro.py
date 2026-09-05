#!/usr/bin/env python3
"""
vit_velcro.py — unified T-VELCRO value-locality pipeline for ViT classifiers.

One CLI, three models behind a small adapter layer:
  --model vit_base      timm vit_base_patch16_224 (no PTQ4ViT dependency)
  --model dinov2_giant  facebook/dinov2-giant-imagenet1k-1-layer (transformers)
  --model dinov3_7b     dimidagd/dinov3-vit7b16-pretrain-lvd1689m-imagenet1k-lc
                        (DINOv3 ViT-7B + ImageNet-1k linear classifier; fp16 on
                        CUDA, fp32 on CPU)

Three modes:
  profile  Welford float64 per-image stats of per-head outputs, captured at the
           input of the attention output projection (attn.proj for timm,
           attention.output.dense for DINOv2, attention.o_proj for DINOv3),
           where heads are still separable.
           Saves per-head variance/mean .npy + per-head entropy JSON.
  freeze   Per block, rank heads (--rank variance|random|entropy) and freeze the
           bottom-T_k. T_k from --heads_per_block or round(--freeze_fraction * H).
           Writes freeze_manifest.json with paths RELATIVE to profile_root.
           Does not load the model (geometry read from the profile meta).
  eval     Mean substitution via forward pre-hook on the output projection
           (output-only, no weight edits), then top-1/top-5 accuracy.

Geometry (blocks/heads/head_dim/seq_len) always comes from the loaded model
config, never from module constants.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODEL_CHOICES = ["vit_base", "dinov2_giant", "dinov3_7b"]
RANK_CHOICES = ["variance", "random", "entropy"]
MODES = ["profile", "freeze", "eval"]


# ==============================================================================
# Model adapters
# ==============================================================================

class Adapter:
    """Uniform view over one ViT classifier.

    proj(b) returns the module whose INPUT is the concatenated per-head
    attention output (B, N, H*D), i.e. before heads are mixed.
    """

    def __init__(self, key, model_id, model, transform,
                 num_blocks, num_heads, head_dim, seq_len, proj_fn):
        self.key = key
        self.model_id = model_id
        self.model = model
        self.transform = transform
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self._proj_fn = proj_fn

    def proj(self, block_idx):
        return self._proj_fn(block_idx)

    def geometry(self) -> dict:
        return {
            "model_key": self.key,
            "model": self.model_id,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
        }


class _ProcessorTransform:
    """Wraps a HuggingFace AutoImageProcessor as a torchvision transform."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def _build_vit_base() -> Adapter:
    import timm
    model_id = "vit_base_patch16_224"
    model = timm.create_model(model_id, pretrained=True)
    model.eval()
    try:
        cfg = timm.data.resolve_model_data_config(model)
    except AttributeError:  # older timm
        cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg, is_training=False)

    num_blocks = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.embed_dim // num_heads
    seq_len = model.patch_embed.num_patches + getattr(model, "num_prefix_tokens", 1)

    return Adapter("vit_base", model_id, model, transform,
                   num_blocks, num_heads, head_dim, seq_len,
                   proj_fn=lambda b: model.blocks[b].attn.proj)


def _build_dinov2_giant() -> Adapter:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model_id = "facebook/dinov2-giant-imagenet1k-1-layer"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()

    cfg = model.config
    num_blocks = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads
    img_size = processor.size.get("height", processor.size.get("shortest_edge", 518))
    seq_len = (img_size // cfg.patch_size) ** 2 + 1

    return Adapter("dinov2_giant", model_id, model, _ProcessorTransform(processor),
                   num_blocks, num_heads, head_dim, seq_len,
                   proj_fn=lambda b: model.dinov2.encoder.layer[b].attention.output.dense)


def _build_dinov3_7b() -> Adapter:
    import torch
    from transformers import DINOv3ViTImageProcessor
    from surgery_dinov3 import DINOV3_7B_LC_ID, DINOv3ViTForImageClassification

    # fp16 on GPU (6.7B params must fit in 24GB), fp32 on CPU.
    # ponytail: weights stage through host RAM (~13.4GB) before build_adapter's
    # .cuda(); add accelerate + device_map="cuda" here if the host is RAM-tight.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = DINOv3ViTForImageClassification.from_pretrained(DINOV3_7B_LC_ID, dtype=dtype)
    model.eval()

    cfg = model.config
    num_blocks = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads
    img_size = cfg.image_size
    seq_len = (img_size // cfg.patch_size) ** 2 + 1 + cfg.num_register_tokens
    # the LC repo ships no preprocessor_config.json; use the DINOv3 defaults
    # (bilinear resize, ImageNet mean/std) at the config's image size
    processor = DINOv3ViTImageProcessor(size={"height": img_size, "width": img_size})

    return Adapter("dinov3_7b", DINOV3_7B_LC_ID, model, _ProcessorTransform(processor),
                   num_blocks, num_heads, head_dim, seq_len,
                   proj_fn=lambda b: model.dinov3.model.layer[b].attention.o_proj)


ADAPTER_BUILDERS = {"vit_base": _build_vit_base, "dinov2_giant": _build_dinov2_giant,
                    "dinov3_7b": _build_dinov3_7b}


def build_adapter(key: str) -> Adapter:
    import torch
    adapter = ADAPTER_BUILDERS[key]()
    if torch.cuda.is_available():
        adapter.model.cuda()
        print(f"  Device: {torch.cuda.get_device_name()}")
    else:
        print("  Device: CPU (WARNING: will be slow)")
    print(f"  Geometry: {adapter.num_blocks} blocks x {adapter.num_heads} heads "
          f"x {adapter.head_dim} head_dim, seq_len={adapter.seq_len}")
    return adapter


# ==============================================================================
# Welford accumulator (float64, per image)
# ==============================================================================

class WelfordAccumulator:
    """Numerically stable online variance (float64)."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None
        self.entropy_sum = 0.0
        self.entropy_count = 0

    def update(self, x):
        arr = x.double()
        if self.mean is None:
            import torch
            self.mean = torch.zeros_like(arr)
            self.M2 = torch.zeros_like(arr)
        self.count += 1
        delta = arr - self.mean
        self.mean += delta / self.count
        delta2 = arr - self.mean
        self.M2 += delta * delta2

    def update_entropy(self, x):
        import torch
        magnitudes = torch.sqrt(torch.sum(x.float() ** 2, dim=-1) + 1e-10)
        probs = magnitudes / (torch.sum(magnitudes) + 1e-10)
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        self.entropy_sum += entropy
        self.entropy_count += 1

    def variance(self) -> Optional[np.ndarray]:
        if self.count < 2 or self.M2 is None:
            return None
        return (self.M2 / (self.count - 1)).numpy().astype(np.float32)

    def get_mean(self) -> Optional[np.ndarray]:
        if self.mean is None:
            return None
        return self.mean.numpy().astype(np.float32)

    def get_entropy(self) -> Optional[float]:
        if self.entropy_count == 0:
            return None
        return self.entropy_sum / self.entropy_count


# ==============================================================================
# Profiling hooks (per-head, at output-projection input)
# ==============================================================================

class HeadProfiler:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.handles: List = []
        self.head_acc: Dict[Tuple[int, int], WelfordAccumulator] = {}

    def register(self):
        for b in range(self.adapter.num_blocks):
            self.handles.append(
                self.adapter.proj(b).register_forward_hook(self._make_hook(b))
            )

    def _make_hook(self, block_idx: int):
        H, D = self.adapter.num_heads, self.adapter.head_dim

        def hook(module, inputs, output):
            x = inputs[0].detach().cpu()  # (B, N, H*D), heads still separable
            B, N, C = x.shape
            x = x.view(B, N, H, D)
            for bi in range(B):
                for h in range(H):
                    key = (block_idx, h)
                    if key not in self.head_acc:
                        self.head_acc[key] = WelfordAccumulator()
                    head_data = x[bi, :, h, :]
                    self.head_acc[key].update(head_data)
                    self.head_acc[key].update_entropy(head_data)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==============================================================================
# Mean-head bypass (output-only forward pre-hook, no weight edits)
# ==============================================================================

class MeanHeadPatcher:
    def __init__(self, adapter: Adapter, frozen_heads: Dict[Tuple[int, int], np.ndarray]):
        self.adapter = adapter
        self.hooks: List = []
        self.block_frozen: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        for (b, h), mean_arr in frozen_heads.items():
            self.block_frozen[b][h] = mean_arr

    def apply(self):
        import torch
        device = next(self.adapter.model.parameters()).device
        dtype = next(self.adapter.model.parameters()).dtype

        for block_idx, head_means in sorted(self.block_frozen.items()):
            mean_tensors = {
                h: torch.from_numpy(m.astype(np.float32)).to(device=device, dtype=dtype)
                for h, m in head_means.items()
            }
            frozen_sorted = sorted(head_means.keys())
            self._add_hook(self.adapter.proj(block_idx), frozen_sorted, mean_tensors)
            print(f"  [mean-head] block {block_idx}: frozen heads {frozen_sorted} "
                  f"({self.adapter.num_heads - len(frozen_sorted)} active)")

    def _add_hook(self, proj_module, frozen_sorted, mean_tensors):
        H, D = self.adapter.num_heads, self.adapter.head_dim

        def hook(module, inputs):
            x = inputs[0]
            B, N, C = x.shape
            reshaped = x.view(B, N, H, D).clone()
            for h in frozen_sorted:
                mean_h = mean_tensors[h]
                if mean_h.shape[0] != N:
                    mean_h = mean_h.mean(dim=0, keepdim=True).expand(N, -1)
                reshaped[:, :, h, :] = mean_h.unsqueeze(0)
            return (reshaped.view(B, N, C),)

        self.hooks.append(proj_module.register_forward_pre_hook(hook))

    def restore(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ==============================================================================
# Data loading (ImageFolder + seeded subset)
# ==============================================================================

def load_dataset(img_dir: str, transform, max_images: Optional[int], seed: int):
    from torch.utils.data import Subset
    from torchvision.datasets import ImageFolder

    val_dir = os.path.join(img_dir, "val")
    root = val_dir if os.path.isdir(val_dir) else img_dir
    dataset = ImageFolder(root, transform=transform)
    available = len(dataset)
    if max_images is not None and max_images < available:
        # Seeded uniform subset (not first-N in folder order, unlike legacy)
        indices = sorted(random.Random(seed).sample(range(available), max_images))
        dataset = Subset(dataset, indices)
        print(f"    Using {max_images}/{available} images (seed={seed})")
    else:
        print(f"    Using all {available} images")
    return dataset


def build_imagenet_label_mapping(synset_words_path: str, class_names: List[str]) -> Dict[int, int]:
    """{local ImageFolder idx -> ImageNet class idx} via synset_words.txt."""
    folder_to_imagenet: Dict[str, int] = {}
    with open(synset_words_path) as f:
        for imagenet_idx, line in enumerate(f):
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            synonyms = [s.strip() for s in parts[1].split(',')]
            joined = '_'.join(s.replace(' ', '_') for s in synonyms)
            first = synonyms[0].replace(' ', '_')
            for key in (joined, joined.lower(), first, first.lower(), parts[0]):
                folder_to_imagenet.setdefault(key, imagenet_idx)

    mapping: Dict[int, int] = {}
    missing = []
    for local_idx, cls_name in enumerate(class_names):
        imagenet_idx = folder_to_imagenet.get(
            cls_name, folder_to_imagenet.get(cls_name.lower()))
        if imagenet_idx is None:
            missing.append(cls_name)
            mapping[local_idx] = local_idx
        else:
            mapping[local_idx] = imagenet_idx

    if missing:
        print(f"  [WARN] {len(missing)} class(es) not in synset_words.txt "
              f"(using local idx): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return mapping


def apply_imagenet_label_mapping(dataset, synset_words_path: Optional[str]) -> None:
    from torch.utils.data import Subset
    if not synset_words_path:
        print("  [WARN] --synset_words not set: accuracy uses local folder indices")
        return
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    mapping = build_imagenet_label_mapping(synset_words_path, base.classes)
    base.target_transform = lambda y, m=mapping: m.get(y, y)
    print(f"  [label mapping] {len(base.classes)} classes mapped via {synset_words_path}")


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(model, dataset, batch_size: int, num_workers: int):
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    total_batches = len(loader)
    top1_correct = top5_correct = total = 0

    with torch.inference_mode():
        for i, (inp, target) in enumerate(loader):
            inp = inp.to(device, dtype=dtype, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(inp)
            logits = output.logits if hasattr(output, "logits") else output

            _, pred = logits.max(1)
            top1_correct += pred.eq(target).sum().item()
            _, pred5 = logits.topk(5, 1, True, True)
            top5_correct += pred5.eq(target.unsqueeze(1).expand_as(pred5)).sum().item()
            total += target.size(0)

            if (i + 1) % 5 == 0:
                print(f"    batch {i+1}/{total_batches}  "
                      f"top1={top1_correct/total*100:.1f}%  "
                      f"top5={top5_correct/total*100:.1f}%")

    return top1_correct, top5_correct, total


# ==============================================================================
# Profile outputs
# ==============================================================================

def save_profile(group: str, profiler: HeadProfiler, out_dir: Path):
    head_dir = out_dir / "head_variance" / group
    head_dir.mkdir(parents=True, exist_ok=True)
    entropy_data = {}

    for (b, h), acc in profiler.head_acc.items():
        var = acc.variance()
        if var is None:
            continue
        np.save(head_dir / f"{group}_block{b}_head{h}_variance.npy", var.ravel())
        mean = acc.get_mean()
        if mean is not None:
            np.save(head_dir / f"{group}_block{b}_head{h}_mean.npy", mean)
        ent = acc.get_entropy()
        if ent is not None:
            entropy_data[f"block{b}_head{h}"] = round(ent, 6)

    if entropy_data:
        ent_dir = out_dir / "entropy" / group
        ent_dir.mkdir(parents=True, exist_ok=True)
        with open(ent_dir / f"{group}_entropy.json", "w") as f:
            json.dump(entropy_data, f, indent=2)

    print(f"  [profile] Saved {len(profiler.head_acc)} head variance/mean files "
          f"+ entropy for '{group}'")


# ==============================================================================
# Freeze: T_k resolution + ranking
# ==============================================================================

def resolve_tk(heads_per_block: Optional[int], freeze_fraction: Optional[float],
               num_heads: int) -> int:
    if heads_per_block is not None:
        t_k = heads_per_block
    else:
        t_k = round(freeze_fraction * num_heads)
        print(f"  T_k = round({freeze_fraction} * {num_heads}) = {t_k} "
              f"(effective fraction {t_k / num_heads:.3f})")
    return max(0, min(t_k, num_heads))


def warn_tk_collisions(out_dir: Path, profile_group: str, t_k: int,
                       freeze_fraction: Optional[float]):
    """Warn when a sibling operating point (different requested fraction) resolved
    to the same T_k — the legacy sweep silently ran duplicate points
    (int(0.25*12) == int(0.33*12) == 3)."""
    if freeze_fraction is None:
        return
    try:
        sweep_root = out_dir.resolve().parent.parent
        own = out_dir.resolve() / "freeze_manifest.json"
        for mp in sweep_root.glob("*/freeze_*/freeze_manifest.json"):
            if mp.resolve() == own:
                continue
            with open(mp) as f:
                m = json.load(f)
            other_frac = m.get("freeze_fraction")
            if (m.get("profile_group") == profile_group and m.get("t_k") == t_k
                    and other_frac is not None and other_frac != freeze_fraction):
                print(f"  [WARN] freeze_fraction={freeze_fraction} and "
                      f"{other_frac} ({mp}) both resolve to T_k={t_k}: "
                      f"duplicate operating point")
    except OSError:
        pass


def rank_block_heads(block: int, heads: List[int], method: str,
                     spec_dir: Path, group: str,
                     entropy_map: Optional[dict], seed: int) -> Dict[int, float]:
    """Lower rank value = more compressible = frozen first."""
    if method == "variance":
        return {h: float(np.sum(np.load(
            spec_dir / f"{group}_block{block}_head{h}_variance.npy")))
            for h in heads}
    if method == "entropy":
        ranks = {}
        for h in heads:
            key = f"block{block}_head{h}"
            if key not in entropy_map:
                raise ValueError(f"Entropy missing for {key}; re-run profile")
            ranks[h] = float(entropy_map[key])
        return ranks
    # random: seeded per-block shuffle
    order = list(heads)
    random.Random(f"{seed}:{block}").shuffle(order)
    return {h: float(order.index(h)) for h in heads}


def freeze_mode(args):
    profile_root = Path(args.profile_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = profile_root / "meta" / "run_info.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — freeze reads model geometry from the "
            f"profile meta; run profile mode with this tool first")
    with open(meta_path) as f:
        profile_meta = json.load(f)
    if profile_meta.get("model_key") not in (None, args.model):
        raise ValueError(f"Profile was built with model "
                         f"'{profile_meta['model_key']}' but --model is '{args.model}'")
    num_blocks = profile_meta["num_blocks"]
    num_heads = profile_meta["num_heads"]

    t_k = resolve_tk(args.heads_per_block, args.freeze_fraction, num_heads)
    warn_tk_collisions(out_dir, args.profile_group, t_k, args.freeze_fraction)

    spec_dir = profile_root / "head_variance" / args.profile_group
    entropy_map = None
    if args.rank == "entropy":
        ent_path = (profile_root / "entropy" / args.profile_group /
                    f"{args.profile_group}_entropy.json")
        with open(ent_path) as f:
            entropy_map = json.load(f)

    print("\n" + "=" * 70)
    print(f"FREEZE MODE — T-VELCRO ({profile_meta.get('model_key', '?')}, "
          f"rank={args.rank}, T_k={t_k}/{num_heads})")
    print("=" * 70)

    frozen_entries = []
    t_k_per_block = {}
    n_total = 0

    for b in range(num_blocks):
        heads = [h for h in range(num_heads)
                 if (spec_dir / f"{args.profile_group}_block{b}_head{h}_variance.npy").exists()]
        n_total += len(heads)
        if not heads:
            continue
        ranks = rank_block_heads(b, heads, args.rank, spec_dir,
                                 args.profile_group, entropy_map, args.seed)
        frozen = sorted(heads, key=lambda h: ranks[h])[:t_k]
        count = 0
        for h in frozen:
            mean_name = f"{args.profile_group}_block{b}_head{h}_mean.npy"
            if not (spec_dir / mean_name).exists():
                print(f"  [WARN] Mean missing: block {b} head {h}")
                continue
            frozen_entries.append({
                "block": b, "head": h,
                "rank_value": ranks[h],
                # relative to profile_root (requirement d)
                "mean_path": str(Path("head_variance") / args.profile_group / mean_name),
            })
            count += 1
        t_k_per_block[str(b)] = count

    n_frozen = len(frozen_entries)
    manifest = {
        "version": "3.0-tvelcro-unified",
        "model_key": profile_meta.get("model_key"),
        "model": profile_meta.get("model"),
        "profile_root": str(profile_root),
        "profile_group": args.profile_group,
        "rank_method": args.rank,
        "seed": args.seed,
        "heads_per_block": args.heads_per_block,
        "freeze_fraction": args.freeze_fraction,
        "t_k": t_k,
        "t_k_per_block": t_k_per_block,
        "profile_N": profile_meta.get("N", {}).get(args.profile_group),
        "profile_seed": profile_meta.get("seed"),
        "freeze_criterion": f"bottom-T_k per block by {args.rank} rank",
        "freeze_method": "output-only (forward pre-hook mean substitution)",
        "frozen_heads": frozen_entries,
        "summary": {
            "total_heads_checked": n_total,
            "total_frozen": n_frozen,
            "freeze_rate": round(n_frozen / n_total, 4) if n_total else 0.0,
        },
    }
    manifest_path = out_dir / "freeze_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Checked {n_total} heads, froze {n_frozen}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 70)


def load_freeze_manifest(path: str, profile_dir_override: Optional[str]):
    with open(path) as f:
        manifest = json.load(f)
    root = Path(profile_dir_override) if profile_dir_override \
        else Path(manifest["profile_root"])
    frozen_heads = {}
    for e in manifest["frozen_heads"]:
        mean_path = root / e["mean_path"]
        if not mean_path.exists():
            print(f"  [WARN] Missing: {mean_path}")
            continue
        frozen_heads[(e["block"], e["head"])] = np.load(mean_path)
    return frozen_heads, manifest


# ==============================================================================
# Profile / eval driver
# ==============================================================================

def profile_eval_mode(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_names = ([Path(d.rstrip("/")).name for d in args.image_dirs]
                   if args.group_names is None
                   else [x.strip() for x in args.group_names.split(",")])
    if len(group_names) != len(args.image_dirs):
        raise ValueError("group_names count must match image_dirs count")

    mode_label = "PROFILE" if args.mode == "profile" else "EVAL (MEAN-HEAD)"
    print("\n" + "=" * 70)
    print(f"T-VELCRO {mode_label} — {args.model}")
    print("=" * 70)

    adapter = build_adapter(args.model)
    if args.fp16:
        adapter.model.half()

    patcher = None
    manifest = None
    if args.mode == "eval":
        if not args.freeze_manifest:
            raise ValueError("--freeze_manifest required for eval mode")
        frozen_heads, manifest = load_freeze_manifest(
            args.freeze_manifest, args.profile_dir)
        if manifest.get("model_key") not in (None, args.model):
            raise ValueError(
                f"Manifest was built for model '{manifest['model_key']}' "
                f"but --model is '{args.model}'")
        print(f"  Freeze manifest: {args.freeze_manifest} "
              f"({len(frozen_heads)} frozen heads, rank={manifest.get('rank_method')})")
        patcher = MeanHeadPatcher(adapter, frozen_heads)
        patcher.apply()

    all_accuracy = {}
    n_per_group = {}

    for g, img_dir in zip(group_names, args.image_dirs):
        print(f"\n  Group: {g} ({img_dir})")
        profiler = None
        if args.mode == "profile":
            profiler = HeadProfiler(adapter)
            profiler.register()

        dataset = load_dataset(img_dir, adapter.transform, args.max_images, args.seed)
        apply_imagenet_label_mapping(dataset, args.synset_words)

        t0 = time.time()
        top1, top5, total = run_inference(
            adapter.model, dataset, args.batch_size, args.num_workers)
        elapsed = time.time() - t0

        n_per_group[g] = total
        all_accuracy[g] = {
            "top1": round(top1 / total * 100, 2) if total else 0.0,
            "top5": round(top5 / total * 100, 2) if total else 0.0,
            "correct_top1": top1, "correct_top5": top5, "total": total,
            "time_sec": round(elapsed, 2),
        }
        print(f"    Top-1: {all_accuracy[g]['top1']:.2f}%  "
              f"Top-5: {all_accuracy[g]['top5']:.2f}%  ({total} images)")

        if profiler is not None:
            save_profile(g, profiler, out_dir)
            profiler.remove()

    if patcher is not None:
        patcher.restore()

    # Meta (requirement e: N, seed, rank method, T_k per block)
    run_info = {
        "mode": args.mode,
        **adapter.geometry(),
        "groups": group_names,
        "image_dirs": [str(Path(d).resolve()) for d in args.image_dirs],
        "accuracy": all_accuracy,
        "N": n_per_group,
        "seed": args.seed,
        "max_images": args.max_images,
        "batch_size": args.batch_size,
    }
    if manifest is not None:
        n_frozen = len(manifest["frozen_heads"])
        total_heads = adapter.num_blocks * adapter.num_heads
        # per-head attention FLOPs are identical across heads, so the saving
        # ratio equals the frozen-head ratio
        run_info.update({
            "freeze_manifest": str(Path(args.freeze_manifest).resolve()),
            "rank_method": manifest.get("rank_method"),
            "t_k": manifest.get("t_k"),
            "t_k_per_block": manifest.get("t_k_per_block"),
            "profile_group": manifest.get("profile_group"),
            "flop_savings_pct": round(n_frozen / total_heads * 100, 2),
        })

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE ({mode_label})  ->  {args.out_dir}")
    print(f"{'Group':<15} {'Top-1':<10} {'Top-5':<10} {'Images':<10}")
    for g, acc in all_accuracy.items():
        print(f"{g:<15} {acc['top1']:<10.2f} {acc['top5']:<10.2f} {acc['total']:<10}")
    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="T-VELCRO unified ViT value-locality pipeline "
                    "(profile / freeze / eval)")
    ap.add_argument("--model", required=True, choices=MODEL_CHOICES)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--image_dirs", nargs="+", default=None,
                    help="Image dirs (profile/eval); each may contain a val/ subdir")
    ap.add_argument("--group_names", default=None, help="Comma-separated group names")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Seeded uniform subset per group (default: all images)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for image subsetting and random ranking")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--synset_words", default=None,
                    help="synset_words.txt for ImageNet label mapping")
    ap.add_argument("--out_dir", required=True)

    # Freeze
    ap.add_argument("--profile_dir", default=None,
                    help="Profile dir (freeze: required; eval: overrides "
                         "the manifest's recorded profile_root)")
    ap.add_argument("--profile_group", default=None)
    ap.add_argument("--rank", choices=RANK_CHOICES, default="variance")
    ap.add_argument("--heads_per_block", type=int, default=None,
                    help="Exact T_k heads to freeze per block")
    ap.add_argument("--freeze_fraction", type=float, default=None,
                    help="T_k = round(fraction * num_heads)")

    # Eval
    ap.add_argument("--freeze_manifest", default=None)
    ap.add_argument("--fp16", action="store_true",
                    help="Run the model in float16 (CUDA)")

    args = ap.parse_args()

    if args.mode == "freeze":
        if not args.profile_dir or not args.profile_group:
            raise ValueError("freeze mode requires --profile_dir and --profile_group")
        if args.heads_per_block is None and args.freeze_fraction is None:
            raise ValueError("freeze mode requires --heads_per_block or --freeze_fraction")
        freeze_mode(args)
        return

    if not args.image_dirs:
        raise ValueError("--image_dirs required for profile/eval modes")
    profile_eval_mode(args)


if __name__ == "__main__":
    main()
