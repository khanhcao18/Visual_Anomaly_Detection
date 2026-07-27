# Visual Anomaly Detection

An end-to-end machine learning system for detecting defective objects using convolutional autoencoders, pretrained CNN features, and classical anomaly-detection algorithms.

## Highlights

- Trains models from raw image data
- Compares deep learning and classical ML approaches
- Automatically selects anomaly thresholds
- Produces defect-localization heatmaps
- Includes an interactive web demo and REST API

## Dataset Layout

Place images in this structure:

```text
data/raw/
├── train/
│   └── normal/
├── test/
│   ├── normal/
│   └── abnormal/
```

The autoencoder is trained only on `train/normal`. Evaluation uses both normal and abnormal images.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Autoencoder

```bash
python train.py --config configs/autoencoder.yaml
```

This saves:

- `models/autoencoder.pt`
- `models/autoencoder_threshold.json`
- `reports/autoencoder_history.csv`

## Evaluate

```bash
python evaluate.py --config configs/autoencoder.yaml
```

The evaluation reports precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, and a threshold sweep.

## Predict One Image

```bash
python predict.py path/to/image.png
```

Example output:

```text
Prediction: Defective
Anomaly score: 0.083000
Threshold: 0.041000
Confidence: High
Heatmap: /absolute/path/reports/figures/prediction_heatmap.png
```

## Compare Baselines

Train ResNet feature baselines:

```bash
python train_baseline.py --model-type isolation_forest
python train_baseline.py --model-type one_class_svm
python train_baseline.py --model-type pca
```

The pretrained ResNet extracts visual embeddings, then the anomaly detector learns the normal feature distribution.

## Web Demo

```bash
python app/gradio_app.py
```

## REST API

```bash
uvicorn app.api:app --reload
```

Then send an image to `POST /predict`.

## Docker

```bash
docker build -t visual-anomaly-detection .
docker run -p 7860:7860 visual-anomaly-detection
```

## Repository Structure

```text
configs/                 Reproducible experiment settings
src/data/                Dataset and preprocessing code
src/models/              Autoencoder and feature extractor
src/training/            Training loop and losses
src/evaluation/          Metrics and threshold selection
src/visualization/       Reconstruction-error heatmaps
app/                     Gradio and FastAPI apps
tests/                   Unit tests
models/                  Saved checkpoints
reports/                 Metrics and generated figures
```
