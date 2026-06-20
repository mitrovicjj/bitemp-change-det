# Deep Learning for Infrastructure Change Detection

A geospatial deep learning project for detecting infrastructure changes in urban and industrial areas using bi-temporal Sentinel-2 imagery and the OSCD benchmark.

## Functionalities

- Detects pixel-level changes between two Sentinel-2 satellite images (T1/T2)
- Trains and evaluates multiple U-Net architectures for binary change mask prediction
- Supports both the public OSCD benchmark and a custom local dataset
- Tracks experiments with MLflow (metrics, configs, visualizations)

## Datasets

**OSCD** — Onera Satellite Change Detection benchmark; 24 bi-temporal Sentinel-2 image pairs from 14 cities (2015–2018), pixel-level binary change labels. RGB variant used (via Hugging Face) to align with ImageNet-pretrained encoder.

**Local dataset** — Custom bi-temporal Sentinel-2 scenes covering infrastructure changes in Serbia. Scenes acquired via Copernicus, annotated in QGIS. Used for domain transfer evaluation and fine-tuning.

## Models

| Model | Description |
|---|---|
| Early Fusion U-Net | T1 and T2 concatenated channel-wise as input |
| Dual Stream U-Net | Separate encoders with shared weights, features merged in decoder |
| Pretrained U-Net (ResNet34) | ImageNet-pretrained ResNet34 encoder — best overall results |

## Results

**OSCD benchmark:**

| Model | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| Early Fusion | 0.167 | 0.267 | 0.41 | 0.45 |
| Siamese | 0.174 | 0.258 | 0.54 | 0.22 |
| Pretrained ResNet34 | 0.256 | 0.386 | 0.45 | 0.36 |

**Loss function ablation (Pretrained model):**

| Loss | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| BCE + Dice (pos_weight) | 0.257 | 0.387 | 0.454 | 0.362 |
| Dice only | 0.201 | 0.320 | 0.426 | 0.338 |
| Focal + Dice | 0.219 | 0.340 | 0.358 | 0.392 |

**Local domain (zero-shot to fine-tuned):**

| Stage | IoU | Dice |
|---|---|---|
| Zero-shot | 0.052 | 0.094 |
| Fine-tuned | 0.145 | 0.238 |

Experiment tracking is handled via MLflow. Results (metrics, configs, prediction visualizations) are logged locally.

## Tech stack

- **PyTorch** + `segmentation_models_pytorch`
- **QGIS** — annotation, visual inspection, GeoTIFF validation
- **MLflow** — experiment tracking
- **Hugging Face Datasets** — OSCD RGB variant
- **Copernicus, Google Earth Engine** — data sources

## Limitations & future work

Current limitations include class imbalance (2–5% change pixels), limited spatial resolution of Sentinel-2, and domain shift between OSCD and local scenes. The project is constrained on hyperparameter search and model complexity.

Next steps: transformer-based architectures (BIT), Sentinel-1 SAR fusion, larger local annotated dataset.

## References

- Daudt, R. C. et al. *Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks.* IGARSS.
- Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). *Fully convolutional siamese networks for change detection.* IEEE.

