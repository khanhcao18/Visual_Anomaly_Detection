from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile

from predict import predict_image

app = FastAPI(title="Visual Anomaly Detection API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    with NamedTemporaryFile(delete=False, suffix=Path(file.filename or "image.png").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    return predict_image(
        tmp_path,
        checkpoint_path="models/autoencoder.pt",
        threshold_path="models/autoencoder_threshold.json",
        output_heatmap="reports/figures/api_heatmap.png",
    )
