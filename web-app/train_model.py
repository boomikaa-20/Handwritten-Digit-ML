"""
train_model.py
Run ONCE:  python train_model.py
Optional quick test:  python train_model.py --samples 10000
Loads MNIST, trains PCA+SVM and PCA+Logistic Regression, measures real
train/test metrics, and saves the model and metrics into the models/ folder.
"""

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from digit_recognizer import METRICS_PATH, MODEL_PATH

N_COMPONENTS = 50
SVM_C = 5
TRAIN_ACC_SAMPLE = 10000   # training accuracy is measured on this many training images (saves time)


def evaluate(name, model, X_train, y_train, X_test, y_test, train_time):
    """Compute train and test metrics for one model."""
    print(f"Evaluating {name} ...")
    train_pred = model.predict(X_train[:TRAIN_ACC_SAMPLE])
    t0 = time.time()
    test_pred = model.predict(X_test)
    predict_time = time.time() - t0

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro", zero_division=0)

    result = {
        "train_accuracy": float(accuracy_score(y_train[:TRAIN_ACC_SAMPLE], train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "train_time_sec": float(train_time),
        "test_predict_time_sec": float(predict_time),
    }
    print(f"  train accuracy: {result['train_accuracy']:.4f} | "
          f"TEST accuracy: {result['test_accuracy']:.4f}")
    return result, test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=0,
                        help="Use only this many training images (0 = all 60,000). Handy for a quick test.")
    args = parser.parse_args()

    print("Loading MNIST (first run downloads it, needs internet)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.float32) / 255.0      # normalise pixels to 0..1
    y = y.astype(int)

    # Standard official split: first 60,000 = training, last 10,000 = testing
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    if args.samples > 0:
        X_train, y_train = X_train[:args.samples], y_train[:args.samples]
    print(f"Training images: {len(X_train)} | Test images: {len(X_test)}")

    # ---------- Main model: PCA + SVM ----------
    svm_model = Pipeline([
        ("pca", PCA(n_components=N_COMPONENTS, random_state=42)),
        ("svm", SVC(kernel="rbf", C=SVM_C, gamma="scale")),
    ])
    print("Training PCA + SVM (this can take several minutes)...")
    t0 = time.time()
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - t0
    variance_kept = float(svm_model.named_steps["pca"].explained_variance_ratio_.sum())
    print(f"Done in {svm_time:.0f}s. PCA keeps {variance_kept*100:.1f}% of the variance.")

    svm_res, svm_test_pred = evaluate("PCA + SVM", svm_model, X_train, y_train,
                                      X_test, y_test, svm_time)

    # ---------- Comparison model: PCA + Logistic Regression ----------
    lr_model = Pipeline([
        ("pca", PCA(n_components=N_COMPONENTS, random_state=42)),
        ("lr", LogisticRegression(max_iter=300)),
    ])
    print("Training PCA + Logistic Regression...")
    t0 = time.time()
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - t0
    lr_res, _ = evaluate("PCA + Logistic Regression", lr_model, X_train, y_train,
                         X_test, y_test, lr_time)

    # ---------- Per-class metrics and confusion matrix (SVM, test set) ----------
    p, r, f1, support = precision_recall_fscore_support(
        y_test, svm_test_pred, labels=list(range(10)), zero_division=0)
    per_class = [{"digit": d, "precision": float(p[d]), "recall": float(r[d]),
                  "f1": float(f1[d]), "support": int(support[d])} for d in range(10)]
    cm = confusion_matrix(y_test, svm_test_pred, labels=list(range(10)))

    metrics = {
        "primary_model": "PCA + SVM (RBF)",
        "config": {"pca_components": N_COMPONENTS, "variance_kept": variance_kept,
                   "svm_kernel": "rbf", "svm_C": SVM_C, "svm_gamma": "scale",
                   "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                   "train_accuracy_measured_on": TRAIN_ACC_SAMPLE},
        "models": {"PCA + SVM (RBF)": svm_res, "PCA + Logistic Regression": lr_res},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print("All done. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()