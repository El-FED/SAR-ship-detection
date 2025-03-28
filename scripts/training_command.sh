#!/bin/bash
python train.py \
  --img 256 \
  --batch 8 \
  --epochs 100 \
  --data "/Volumes/Extreme Pro/SARFish_Data/speckle_sarfish.yaml" \
  --weights yolov5s.pt \
  --hyp /Users/facundofed/yolov5/data/hyps/my_hyp_speckle.yaml \
  --name sar_tiled_speckle \
  --patience 10 \
  --device mps
