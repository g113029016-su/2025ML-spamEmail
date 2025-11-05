"""Train baseline spam classifier (Logistic Regression or SVM).

Usage example:
python ml/train_baseline.py --model lr --max_features 5000 --C 1.0
"""
import argparse
import os
import json
from pathlib import Path
import joblib
import pickle  # 新增：支援 pickle 格式
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder  # 新增：Label Encoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

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
    # ✅ 修改：改用 models 目錄
    out_dir = Path(args.output_dir)
    models_dir = out_dir  # 直接使用 models 目錄
    artifacts_dir = out_dir / "artifacts"  # artifacts 放在 models/artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 Spam Classifier Training Script")
    print("=" * 60)
    
    print("\n📥 Loading data...")
    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    print(f"✅ Dataset size: {len(df)}")
    print(f"   - HAM: {(y == 0).sum()}")
    print(f"   - SPAM: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    print(f"\n✂️ Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("\n📊 Building vectorizer...")
    vectorizer = build_vectorizer(args.max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"✅ Vocabulary size: {len(vectorizer.vocabulary_)}")

    print(f"\n🤖 Training model: {args.model}")
    model = build_model(args.model, args.C)
    model.fit(X_train_tfidf, y_train)
    print("✅ Model training completed")

    print("\n📈 Evaluating...")
    metrics = evaluate_model(model, X_test_tfidf, y_test)
    print("\n✅ Model Performance:")
    print(json.dumps(metrics, indent=2))

    # ✅ 修改：儲存為 .pkl 格式（與 app.py 一致）
    model_path = models_dir / "spam_classifier.pkl"  # 改名
    vec_path = models_dir / "vectorizer.pkl"  # 改名
    metrics_path = artifacts_dir / "metrics.json"

    # 使用 pickle 儲存（也可以用 joblib，兩者相容）
    print("\n💾 Saving model artifacts...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")
    
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer saved to: {vec_path}")
    
    # ✅ 新增：建立並儲存 Label Encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['ham', 'spam'])
    label_encoder_path = models_dir / "label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to: {label_encoder_path}")
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")

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
        print("\n📊 Generating confusion matrix...")
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
            disp.plot(ax=ax_cm, cmap='Blues')
            ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            cm_path = artifacts_dir / "confusion_matrix.png"
            fig_cm.savefig(cm_path, bbox_inches="tight", dpi=150)
            plt.close(fig_cm)
            print(f"✅ Confusion matrix saved to: {cm_path}")
        except Exception as e:
            print(f"❌ Failed to create confusion matrix plot: {e}")

        # ROC curve
        if probs is not None:
            print("\n📈 Generating ROC curve...")
            try:
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, color='#e74c3c', lw=2, 
                           label=f"ROC curve (AUC = {roc_auc:.4f})")
                ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, 
                           linestyle='--', label="Random")
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel("False Positive Rate", fontsize=12)
                ax_roc.set_ylabel("True Positive Rate", fontsize=12)
                ax_roc.set_title("ROC Curve", fontsize=14, fontweight='bold')
                ax_roc.legend(loc="lower right", fontsize=10)
                ax_roc.grid(alpha=0.3)
                roc_path = artifacts_dir / "roc_curve.png"
                fig_roc.savefig(roc_path, bbox_inches="tight", dpi=150)
                plt.close(fig_roc)
                print(f"✅ ROC curve saved to: {roc_path}")
            except Exception as e:
                print(f"❌ Failed to create ROC curve plot: {e}")

        # Simple bar chart of primary metrics
        print("\n📊 Generating metrics bar chart...")
        try:
            fig_metrics, axm = plt.subplots(figsize=(8, 6))
            metric_items = ["accuracy", "precision", "recall", "f1"]
            vals = [metrics.get(k, 0.0) or 0.0 for k in metric_items]
            colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
            bars = axm.bar(metric_items, vals, color=colors)
            axm.set_ylim(0, 1.1)
            axm.set_title("Primary Evaluation Metrics", fontsize=14, fontweight='bold')
            axm.set_ylabel("Score", fontsize=12)
            axm.grid(axis='y', alpha=0.3)
            for i, (bar, v) in enumerate(zip(bars, vals)):
                height = bar.get_height()
                axm.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            metrics_path_img = artifacts_dir / "metrics_bar.png"
            fig_metrics.savefig(metrics_path_img, bbox_inches="tight", dpi=150)
            plt.close(fig_metrics)
            print(f"✅ Metrics bar chart saved to: {metrics_path_img}")
        except Exception as e:
            print(f"❌ Failed to create metrics bar chart: {e}")

    except Exception as e:
        print(f"❌ Failed to generate visualizations: {e}")

    print("\n" + "=" * 60)
    print("🎉 Training completed successfully!")
    print("=" * 60)
    print("\n📌 Next steps:")
    print("   1. Run: streamlit run app.py")
    print("   2. Open the Prediction tab")
    print("   3. Test the model with sample messages")
    print("\n✨ Happy classifying! ✨\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline spam classifier")
    parser.add_argument("--model", default="lr", help="Model to train: lr or svm", choices=["lr", "svm"]) 
    parser.add_argument("--max_features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter for models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    # ✅ 修改：預設輸出目錄改為 "models"
    parser.add_argument("--output-dir", default="models", help="Directory to save models/artifacts")
    args = parser.parse_args()
    main(args)
