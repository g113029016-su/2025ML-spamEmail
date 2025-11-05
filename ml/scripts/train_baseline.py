"""Train baseline spam classifier (Logistic Regression or SVM).

Usage example:
python ml/scripts/train_baseline.py --model lr --max_features 5000 --C 1.0
"""
import argparse
import os
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"


def load_data(url=DATA_URL):
    # Dataset has no header; assume first column is label and second column is text
    df = pd.read_csv(url, header=None, names=["label", "text"])
    # Normalize label to binary: spam -> 1, ham -> 0
    df = df.dropna(subset=["text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    return df


def build_vectorizer(max_features: int):
    return TfidfVectorizer(max_features=max_features, stop_words="english")


def build_model(model_name: str, C: float):
    if model_name.lower() in ("lr", "logistic", "logistic_regression"):
        return LogisticRegression(C=C, max_iter=2000)
    else:
        # SVM with probability for confidence scores
        return SVC(C=C, probability=True)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        probs = None
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if probs is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def main(args):
    out_dir = Path(args.output_dir)
    models_dir = out_dir / "models"
    artifacts_dir = out_dir / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    print(f"Dataset size: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    print("Building vectorizer...")
    vectorizer = build_vectorizer(args.max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Training model: {args.model}")
    model = build_model(args.model, args.C)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    metrics = evaluate_model(model, X_test_tfidf, y_test)
    print(json.dumps(metrics, indent=2))

    # Save artifacts
    model_path = models_dir / "model.joblib"
    vec_path = models_dir / "vectorizer.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Generate and save example visualizations for Streamlit to consume
    try:
        # Predictions and probabilities (if available)
        y_pred = model.predict(X_test_tfidf)
        probs = None
        try:
            probs = model.predict_proba(X_test_tfidf)[:, 1]
        except Exception:
            probs = None

        # Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
            disp.plot(ax=ax_cm)
            cm_path = artifacts_dir / "confusion_matrix.png"
            fig_cm.savefig(cm_path, bbox_inches="tight")
            plt.close(fig_cm)
            print(f"Saved confusion matrix to {cm_path}")
        except Exception as e:
            print(f"Failed to create confusion matrix plot: {e}")

        # ROC curve
        if probs is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                ax_roc.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.3f})")
                ax_roc.plot([0, 1], [0, 1], "k--", label="Chance")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend(loc="lower right")
                roc_path = artifacts_dir / "roc_curve.png"
                fig_roc.savefig(roc_path, bbox_inches="tight")
                plt.close(fig_roc)
                print(f"Saved ROC curve to {roc_path}")
            except Exception as e:
                print(f"Failed to create ROC curve plot: {e}")

        # Simple bar chart of primary metrics
        try:
            fig_metrics, axm = plt.subplots(figsize=(6, 4))
            metric_items = ["accuracy", "precision", "recall", "f1"]
            vals = [metrics.get(k, 0.0) or 0.0 for k in metric_items]
            axm.bar(metric_items, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
            axm.set_ylim(0, 1)
            axm.set_title("Primary evaluation metrics")
            for i, v in enumerate(vals):
                axm.text(i, v + 0.01, f"{v:.3f}", ha="center")
            metrics_path_img = artifacts_dir / "metrics_bar.png"
            fig_metrics.savefig(metrics_path_img, bbox_inches="tight")
            plt.close(fig_metrics)
            print(f"Saved metrics bar chart to {metrics_path_img}")
        except Exception as e:
            print(f"Failed to create metrics bar chart: {e}")

    except Exception as e:
        print(f"Failed to generate visualizations: {e}")

    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vec_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline spam classifier")
    parser.add_argument("--model", default="lr", help="Model to train: lr or svm", choices=["lr", "svm"]) 
    parser.add_argument("--max_features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter for models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="ml_output", help="Directory to save models/artifacts")
    args = parser.parse_args()
    main(args)
