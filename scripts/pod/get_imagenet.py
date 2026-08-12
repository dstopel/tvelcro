import os, re
from datasets import load_dataset, Image as HFImage
tok = os.environ["HF_TOKEN"]
ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=tok)
ds = ds.cast_column("image", HFImage(decode=False))
names = ds.features["label"].names
wnid = [n if re.match(r"^n\d{8}$", n) else None for n in names]
if wnid[0] is None:
    with open("/workspace/tvelcro/data/synset_words.txt") as f:
        wnid = sorted(line.split()[0] for line in f if line.strip())
dst = "/workspace/imagenet_val"
n = 0
for ex in ds:
    d = os.path.join(dst, wnid[ex["label"]])
    os.makedirs(d, exist_ok=True)
    name = os.path.basename(ex["image"].get("path") or f"val_{n:08d}.JPEG")
    with open(os.path.join(d, name), "wb") as f:
        f.write(ex["image"]["bytes"])
    n += 1
    if n % 5000 == 0:
        print("saved", n, flush=True)
print("done:", n, "images,", len(os.listdir(dst)), "classes")
