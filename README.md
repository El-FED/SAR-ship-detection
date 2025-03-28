
# SAR Ship Detection – Lightweight Deep Learning for Vessel Detection in SAR Imagery

This repository contains the full implementation of a lightweight deep learning pipeline designed to detect vessels in **Synthetic Aperture Radar (SAR)** imagery, with the goal of addressing **Illegal, Unreported, and Unregulated (IUU) fishing**. It supports efficient training and deployment on consumer-grade hardware (laptops without GPUs).


---

## Project Overview

- **Goal**: Detect ships (especially fishing vessels) from SARFish GRD imagery using lightweight deep learning models.
- **Model**: YOLOv5 (You Only Look Once), optimized for performance on low-resource systems.
- **Dataset**: SARFish (GRD format) – subset of 15 scenes (~20 GB), with 3,600+ confirmed vessels.
- **Target hardware**: MacBook Air M2, 16GB RAM (no dedicated GPU).

---

## Preprocessing and Model Training

The pipeline includes:

- **Normalization** before all filtering steps for efficiency.
- **Lee Filter** for speckle noise reduction.
- **CLAHE** for local contrast enhancement.
- **Tiling** of each large 20,000×20,000 scene into ~256×256 pixel tiles.
- **Anchor box generation** via K-Means clustering.
- **Oversampling** of vessel-containing tiles to handle class imbalance.
- **Symbolic linking** for efficient train/val split without duplication.

The model used is `YOLOv5s`, trained for up to 100 epochs with early stopping. The implementation runs with Apple's MPS backend for MacBooks.

---

## Results

The final model was trained on 195,000 image tiles using YOLOv5 with the SARFish GRD dataset (15 scenes). Performance on a validation set of 2,000 samples is summarized below:

| Metric       | Value     |
|--------------|-----------|
| Precision    | 0.035     |
| Recall       | 0.53      |
| mAP@0.5      | 0.12      |
| mAP@0.5:0.95 | 0.03      |

- **False positives** huge number due to SAR noise and vessel size, most of them under 4 pixels
- **High recall** indicates the model learned to detect most vessels, though at the cost of precision.
- Training was completed in approximately **27 hours** on a MacBook Air M2 (no GPU).
- Preprocessing runtime improved from **15 hours to ~2.5 hours** after optimization.


---

## Resources
- [SARFish Dataset](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish)
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)

---

## Citation

If you use this project or refer to it in your research, please cite:

```
Espina, F. A. (2025). Lightweight Deep Learning for Vessel Detection in SAR Imagery: A Feasibility Study for IUU Fishing Surveillance. MSc Thesis, Gisma University.
```
