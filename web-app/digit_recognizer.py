"""
digit_recognizer.py
Shared helper code used by both the website (app.py) and training (train_model.py).
Contains: model loading, canvas-image preprocessing, and prediction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "digit_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

INK_THRESHOLD = 0.2     # a pixel brighter than this counts as "ink" (0 to 1 scale)
MIN_INK_PIXELS = 40     # fewer ink pixels than this = canvas treated as blank


class ModelNotFoundError(Exception):
    """Raised when the trained model file does not exist yet."""


def load_model():
    """Load the saved PCA + SVM pipeline from disk."""
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            "Model file not found. Please run:  python train_model.py"
        )
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load the evaluation results saved during training. Returns None if missing."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_grayscale(image_data):
    """
    Convert the canvas output (H x W x 4 RGBA array) into a 2-D array in 0..1.
    The canvas uses a BLACK background and WHITE strokes, which matches MNIST
    (white digit on black), so no inversion is needed.
    Transparent pixels are treated as black, so either canvas behaviour works.
    """
    arr = np.asarray(image_data, dtype=np.float32)
    if arr.ndim == 2:
        gray = arr
    elif arr.shape[2] == 4:
        rgb_mean = arr[..., :3].mean(axis=2)
        alpha = arr[..., 3] / 255.0
        gray = rgb_mean * alpha
    else:
        gray = arr[..., :3].mean(axis=2)
    return gray / 255.0


def preprocess_canvas(image_data):
    """
    Turn a canvas drawing into a 28x28 float image (0..1) in MNIST style.
    Steps: grayscale -> find digit bounding box -> crop -> scale longest side
    to 20 px (keeping aspect ratio) -> paste into 28x28 -> centre by centre of mass.
    Returns None if the canvas is blank.
    """
    gray = _to_grayscale(image_data)
    mask = gray > INK_THRESHOLD

    if mask.sum() < MIN_INK_PIXELS:
        return None  # blank (or nearly blank) canvas

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1] + 1
    left, right = cols[0], cols[-1] + 1
    digit = gray[top:bottom, left:right]

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil = Image.fromarray((digit * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    canvas28 = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas28[y0:y0 + new_h, x0:x0 + new_w] = np.asarray(pil, dtype=np.float32) / 255.0

    # MNIST digits are centred by centre of mass; do the same.
    cy, cx = center_of_mass(canvas28)
    canvas28 = shift(canvas28, (13.5 - cy, 13.5 - cx), order=1, mode="constant", cval=0.0)

    return np.clip(canvas28, 0.0, 1.0)


def predict_digit(model, img28):
    """
    Predict the digit from a 28x28 image (0..1).
    Returns a dict with: digit, scores (10 decision scores), top3, margin.
    """
    x = img28.reshape(1, -1)                    # same flattened format as training
    digit = int(model.predict(x)[0])
    scores = model.decision_function(x)[0]      # one score per digit 0..9

    order = np.argsort(scores)[::-1]
    top3 = [(int(d), float(scores[d])) for d in order[:3]]
    margin = float(scores[order[0]] - scores[order[1]])

    return {"digit": digit, "scores": scores, "top3": top3, "margin": margin}