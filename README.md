# Pollen Counter 🌻🔬

A CNN-based regression model that counts pollen grains in microscope images using transfer learning.

## Overview

This project uses **EfficientNetB0** (pretrained on ImageNet) to predict the number of pollen grains in microscope images. The model is trained with data augmentation and two-phase training (frozen backbone → fine-tuning), and exports to TensorFlow.js for web deployment.

## Files

| File | Description |
|------|-------------|
| `Pollen_Counter_v2.py` | **Latest version** — Transfer learning, callbacks, evaluation plots |
| `Pollen_Counter_v1.ipynb` | Original Colab notebook (20 epochs, 360 images) |
| `5_20_24_Pollen_Counter_v0.ipynb` | Earlier prototype (10 epochs, 279 images) |

## Setup

### Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy, matplotlib

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Data

1. Place pollen microscope images in a folder (e.g., `Pollen Training Images/`)
2. Create a CSV file (`image_counts.csv`) with columns:
   - `filename` — image filename
   - `count` — number of pollen grains in the image

### Running

**Google Colab:**
1. Upload `Pollen_Counter_v2.py` to Colab
2. Ensure your `Pollen Training Images` folder and `image_counts.csv` are in Google Drive
3. Run: `%run Pollen_Counter_v2.py`

**Local:**
1. Update `DATA_DIR` and `CSV_FILE` paths in the configuration section
2. Run: `python Pollen_Counter_v2.py`

## Model Architecture

- **Backbone**: EfficientNetB0 (pretrained on ImageNet)
- **Head**: GlobalAveragePooling2D → Dense(256) → Dropout(0.4) → Dense(64) → Dropout(0.2) → Dense(1)
- **Loss**: Mean Squared Error (regression)
- **Metrics**: MAE (mean absolute error in pollen count units)

## Training Strategy

1. **Phase 1** — Frozen backbone, train head only (lr=1e-3, up to 50 epochs)
2. **Phase 2** — Unfreeze top 30% of backbone, fine-tune (lr=1e-5, up to 30 epochs)
3. Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Web Deployment

The model automatically exports to TensorFlow.js format for browser-based inference.
