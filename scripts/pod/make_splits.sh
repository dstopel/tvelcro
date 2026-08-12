#!/bin/bash
set -uo pipefail
D=/workspace/tvelcro/data; SEED=0; NP=300; NE=300
split_domain() {
  local name=$1 findexpr=$2
  local prof=$D/image_dirs/${name}_profile eval_=$D/image_dirs/${name}_eval
  [ -d "$prof" ] && [ "$(find $prof -type f | wc -l)" -gt 0 ] && { echo "$name exists"; return; }
  mkdir -p "$prof" "$eval_"
  local shuffled=$(eval "$findexpr" | sort | awk -v s=$SEED "{print s\"_\"NR\"_\"\$0, \$0}" | sort -t" " -k1,1 | awk "{print \$2}")
  local n=0
  while IFS= read -r img; do
    n=$((n+1))
    if [ $n -le $NP ]; then ln "$img" "$prof/$(basename "$img")" 2>/dev/null || cp "$img" "$prof/"
    elif [ $n -le $((NP+NE)) ]; then ln "$img" "$eval_/$(basename "$img")" 2>/dev/null || cp "$img" "$eval_/"
    else break; fi
  done <<< "$shuffled"
  echo "$name: profile=$(ls $prof|wc -l) eval=$(ls $eval_|wc -l)"
}
split_domain kvasir "find $D/kvasir-seg/images -type f -name \"*.jpg\""
split_domain isic "find $D/isic/ISIC2018_Task1-2_Training_Input -type f -name \"*.jpg\" -not -path \"*checkpoints*\""
split_domain busi "find $D/busi/Dataset_BUSI_with_GT -type f -name \"*.png\" -not -name \"*mask*\""
split_domain pcb "find $D/pcb/PCB_DATASET/images -type f \( -name \"*.jpg\" -o -name \"*.JPG\" \)"
split_domain deepglobe "find $D/deepglobe/train -type f -name \"*_sat.jpg\""
