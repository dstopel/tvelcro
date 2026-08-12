#!/bin/bash
# download_data.sh — Download SAM ViT-H checkpoint and COCO val2017
set -e

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading SAM ViT-H checkpoint ==="
if [ ! -f "$DATA_DIR/sam_vit_h_4b8939.pth" ]; then
    wget -O "$DATA_DIR/sam_vit_h_4b8939.pth" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    echo "  Downloaded SAM ViT-H checkpoint"
else
    echo "  SAM checkpoint already exists"
fi

echo "=== Downloading COCO val2017 images ==="
if [ ! -d "$DATA_DIR/val2017" ]; then
    wget -O "$DATA_DIR/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/val2017.zip"
    echo "  Downloaded and extracted COCO val2017"
else
    echo "  COCO val2017 already exists"
fi

echo "=== Downloading COCO annotations ==="
if [ ! -f "$DATA_DIR/annotations/instances_val2017.json" ]; then
    wget -O "$DATA_DIR/annotations_trainval2017.zip" \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/annotations_trainval2017.zip"
    echo "  Downloaded and extracted COCO annotations"
else
    echo "  COCO annotations already exist"
fi

echo ""
echo "=== Done ==="
echo "  SAM checkpoint: $DATA_DIR/sam_vit_h_4b8939.pth"
echo "  COCO images:    $DATA_DIR/val2017/ ($(ls $DATA_DIR/val2017 | wc -l) images)"
echo "  COCO annots:    $DATA_DIR/annotations/instances_val2017.json"
