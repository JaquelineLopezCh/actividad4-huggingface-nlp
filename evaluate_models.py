
"""
Script de soporte para la Actividad 4.
Este archivo documenta de forma reproducible la lógica principal utilizada
para evaluar modelos preentrenados de Hugging Face en una tarea de clasificación
de sentimiento.

Nota:
El notebook de Google Colab contiene la ejecución completa del prototipo.
"""

import time
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def normalize_prediction_label(label):
    label = str(label).upper().strip()

    if label in ["POSITIVE", "LABEL_1", "1"]:
        return 1
    elif label in ["NEGATIVE", "LABEL_0", "0"]:
        return 0
    else:
        return np.nan


def evaluate_model(model_info, dataframe, device=-1):
    model_name = model_info["model_name"]
    short_name = model_info["short_name"]

    classifier = pipeline(
        task=model_info["task"],
        model=model_name,
        tokenizer=model_name,
        device=device
    )

    texts = dataframe["text"].tolist()
    y_true = dataframe["true_label"].tolist()

    predictions = []
    latencies = []

    start_total_time = time.time()

    for text in texts:
        start_time = time.time()

        result = classifier(
            text,
            truncation=True,
            max_length=512
        )[0]

        end_time = time.time()

        predictions.append(normalize_prediction_label(result["label"]))
        latencies.append(end_time - start_time)

    end_total_time = time.time()

    valid_mask = ~pd.isna(pd.Series(predictions))

    y_true_valid = pd.Series(y_true)[valid_mask].astype(int).tolist()
    y_pred_valid = pd.Series(predictions)[valid_mask].astype(int).tolist()

    summary = {
        "model": short_name,
        "huggingface_model": model_name,
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "precision": precision_score(y_true_valid, y_pred_valid, average="binary", zero_division=0),
        "recall": recall_score(y_true_valid, y_pred_valid, average="binary", zero_division=0),
        "f1_score": f1_score(y_true_valid, y_pred_valid, average="binary", zero_division=0),
        "avg_latency_seconds": float(np.mean(latencies)),
        "total_inference_time_seconds": float(end_total_time - start_total_time)
    }

    return summary
