#!/bin/bash
MIOPEN_DISABLE_CACHE=1 ENABLE_XFORMERS=1 HIP_VISIBLE_DEVICES=7 \
python3 scripts/inference.py configs/anhtt/inference.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --prompt "a beautiful vocanic mountain" \
  --reference-path "save/references/sample.jpg"
