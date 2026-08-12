#!/bin/bash
# setup_pod.sh: idempotent RunPod bootstrap for T-VELCRO.
# Safe to re-run: every step checks before it acts. Run from repo root.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
mkdir -p "$DATA_ROOT"

# readiness flags for the final checklist
OK_GPU=no OK_VENV=no OK_SAM=no OK_COCO=no OK_SYNSET=no OK_KAGGLE=no OK_IMAGENET=no

dl() { # dl <url> <dest>: atomic download (tmp file, then mv)
    local url="$1" dest="$2" tmp="$2.part"
    if command -v wget >/dev/null; then wget -q --show-progress -O "$tmp" "$url"
    else curl -fL -o "$tmp" "$url"; fi && mv "$tmp" "$dest"
}

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && OK_GPU=yes
else
    echo "WARNING: nvidia-smi not found; latency/energy experiments need a GPU."
fi

echo "=== Python env ==="
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
if [ ! -f .venv/.deps_stamp ] || [ requirements.txt -nt .venv/.deps_stamp ]; then
    pip install -q -U pip && CUDA_DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d. -f1) && TORCH_IDX=$([ "${CUDA_DRV:-0}" -ge 580 ] && echo cu130 || echo cu126) && pip install -q --index-url https://download.pytorch.org/whl/$TORCH_IDX torch torchvision && pip install -q -r requirements.txt && touch .venv/.deps_stamp
fi
[ -f .venv/.deps_stamp ] && OK_VENV=yes && echo "venv ready (.venv)"

echo "=== SAM ViT-H checkpoint ==="
SAM_CKPT="$DATA_ROOT/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CKPT" ]; then
    dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "$SAM_CKPT"
fi
[ -f "$SAM_CKPT" ] && OK_SAM=yes && echo "checkpoint: $SAM_CKPT"

echo "=== MobileSAM checkpoint (competitor baseline) ==="
MOBILE_CKPT="$DATA_ROOT/mobile_sam.pt"
if [ ! -f "$MOBILE_CKPT" ]; then
    dl https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt "$MOBILE_CKPT" || \
        echo "MobileSAM download failed; baselines/sam_competitors.py will skip that variant."
fi
# SlimSAM weights auto-download from the HuggingFace hub on first run.

echo "=== COCO val2017 ==="
if [ ! -d "$DATA_ROOT/val2017" ]; then
    dl http://images.cocodataset.org/zips/val2017.zip "$DATA_ROOT/val2017.zip" \
        && unzip -q "$DATA_ROOT/val2017.zip" -d "$DATA_ROOT" && rm "$DATA_ROOT/val2017.zip"
fi
if [ ! -f "$DATA_ROOT/annotations/instances_val2017.json" ]; then
    dl http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$DATA_ROOT/annots.zip" \
        && unzip -q "$DATA_ROOT/annots.zip" -d "$DATA_ROOT" && rm "$DATA_ROOT/annots.zip"
fi
[ -d "$DATA_ROOT/val2017" ] && [ -f "$DATA_ROOT/annotations/instances_val2017.json" ] \
    && OK_COCO=yes && echo "COCO ready under $DATA_ROOT"

echo "=== ImageNet synset_words.txt (ViT label mapping) ==="
if [ ! -f "$DATA_ROOT/synset_words.txt" ]; then
    if dl http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz "$DATA_ROOT/ilsvrc12_aux.tar.gz"; then
        tar -xzf "$DATA_ROOT/ilsvrc12_aux.tar.gz" -C "$DATA_ROOT" synset_words.txt
        rm "$DATA_ROOT/ilsvrc12_aux.tar.gz"
    else
        echo "WARNING: could not fetch caffe_ilsvrc12.tar.gz; place synset_words.txt in $DATA_ROOT manually."
    fi
fi
[ -f "$DATA_ROOT/synset_words.txt" ] && OK_SYNSET=yes

echo "=== Kaggle domain datasets (SAM) ==="
# name:slug pairs used by the SAM domain experiments
KAGGLE_SETS="kvasir:debeshjha1/kvasirseg isic:tschandl/isic2018-challenge-task1-data-segmentation busi:aryashah2k/breast-ultrasound-images-dataset pcb:akhatova/pcb-defects deepglobe:balraj98/deepglobe-road-extraction-dataset"
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    OK_KAGGLE=yes
    for pair in $KAGGLE_SETS; do
        name="${pair%%:*}" slug="${pair#*:}"
        if [ -d "$DATA_ROOT/$name" ]; then
            echo "$name: already present"
        elif kaggle datasets download -d "$slug" -p "$DATA_ROOT" --unzip -o \
                && mkdir -p "$DATA_ROOT/$name"; then
            echo "$name: downloaded ($slug); files under $DATA_ROOT (move into $DATA_ROOT/$name if the archive did not create it)"
        else
            echo "$name: download FAILED ($slug)"; OK_KAGGLE=partial
        fi
    done
else
    echo "KAGGLE_USERNAME / KAGGLE_KEY not set. To fetch the domain datasets manually:"
    echo "  export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>   # from kaggle.com -> Account -> Create API Token"
    for pair in $KAGGLE_SETS; do
        echo "  kaggle datasets download -d ${pair#*:} -p $DATA_ROOT/${pair%%:*} --unzip"
    done
fi

echo "=== ImageNet val (ViT) ==="
# Cannot be auto-downloaded (license). Expect a val-by-class tree:
#   $IMAGENET_DIR/<class_dir>/*.JPEG
if [ -z "${IMAGENET_DIR:-}" ]; then
    echo "ERROR: IMAGENET_DIR is not set."
    echo "  Download ILSVRC2012 val (image-net.org), restructure into one dir per class, then:"
    echo "  export IMAGENET_DIR=/path/to/val_by_class && bash setup_pod.sh"
elif [ ! -d "$IMAGENET_DIR" ]; then
    echo "ERROR: IMAGENET_DIR=$IMAGENET_DIR is not a directory."
elif [ "$(find "$IMAGENET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -lt 2 ]; then
    echo "ERROR: IMAGENET_DIR=$IMAGENET_DIR has no class subdirectories."
    echo "  Expected layout: \$IMAGENET_DIR/<class_dir>/*.JPEG (val images grouped by class)."
else
    OK_IMAGENET=yes
    echo "IMAGENET_DIR ok: $(find "$IMAGENET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) class dirs"
fi

echo ""
echo "=== Checklist ==="
echo "  GPU:                 $OK_GPU"
echo "  venv + deps:         $OK_VENV"
echo "  SAM checkpoint:      $OK_SAM"
echo "  COCO val2017:        $OK_COCO"
echo "  synset_words.txt:    $OK_SYNSET"
echo "  Kaggle domains:      $OK_KAGGLE"
echo "  ImageNet val tree:   $OK_IMAGENET"
echo ""
echo "Suggested first command (SAM path, needs only the items above marked yes):"
echo "  PYTHONPATH=. bash experiments/sam/exp_main_sweep.sh"
echo "ViT path: first build domain groups, then run the ViT sweep:"
echo "  PYTHONPATH=. python vit/create_groups.py --src \$IMAGENET_DIR --dst $DATA_ROOT/vit_groups"
echo "  PYTHONPATH=. bash experiments/vit/exp_sweep.sh"
