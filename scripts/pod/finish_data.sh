#!/bin/bash
set -uo pipefail
source /workspace/.secrets.env
cd /workspace/tvelcro
source .venv/bin/activate
D=/workspace/tvelcro/data
echo "=== COCO (resume) ==="
mkdir -p $D/coco && cd $D/coco
wget -c -q http://images.cocodataset.org/zips/val2017.zip && unzip -q -n val2017.zip
wget -c -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q -n annotations_trainval2017.zip
echo "coco done: $(ls val2017 2>/dev/null | wc -l) images"
echo "=== Kaggle domains (KGAT token) ==="
export KAGGLE_API_TOKEN
for spec in "debeshjha1/kvasirseg kvasir" "tschandl/isic2018-challenge-task1-data-segmentation isic" "aryashah2k/breast-ultrasound-images-dataset busi" "akhatova/pcb-defects pcb" "balraj98/deepglobe-road-extraction-dataset deepglobe"; do
  set -- $spec
  [ -d "$D/$2" ] && { echo "$2 exists"; continue; }
  kaggle datasets download -d "$1" -p "$D/$2" --unzip && echo "$2 ok" || echo "$2 FAILED"
done
echo "=== ImageNet val from HF ==="
if [ ! -d /workspace/imagenet_val ]; then
  cd /workspace
  python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='ILSVRC/imagenet-1k', filename='data/val_images.tar.gz', repo_type='dataset', token='', local_dir='/workspace/_hf')
print(p)"
  mkdir -p imagenet_val_flat && tar xzf _hf/data/val_images.tar.gz -C imagenet_val_flat
  python - <<PY
import os, re, shutil
src, dst = "/workspace/imagenet_val_flat", "/workspace/imagenet_val"
os.makedirs(dst, exist_ok=True)
n = 0
for f in os.listdir(src):
    m = re.search(r"_(n\d{8})\.JPEG$", f)
    if m:
        d = os.path.join(dst, m.group(1)); os.makedirs(d, exist_ok=True)
        shutil.move(os.path.join(src, f), os.path.join(d, f)); n += 1
print("sorted", n, "images into", len(os.listdir(dst)), "classes")
PY
  rm -rf /workspace/imagenet_val_flat /workspace/_hf
fi
echo "=== ALL DATA DONE ==="
