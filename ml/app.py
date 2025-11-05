import subprocess
import string
import re
import streamlit as st
import joblib
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODELS_DIR = Path("ml_output/models")
ARTIFACTS_DIR = Path("ml_output/artifacts")

st.set_page_config(page_title="Spam Classifier Demo", layout="wide")

st.title("Spam Email/SMS Classification Demo")

# Load model if present
model = None
vectorizer = None
metrics = None

if MODELS_DIR.joinpath("model.joblib").exists() and MODELS_DIR.joinpath("vectorizer.joblib").exists():
    try:
        model = joblib.load(MODELS_DIR.joinpath("model.joblib"))
        vectorizer = joblib.load(MODELS_DIR.joinpath("vectorizer.joblib"))
    except Exception as e:
        st.warning(f"Failed to load model/vectorizer: {e}")

if ARTIFACTS_DIR.joinpath("metrics.json").exists():
    try:
        with open(ARTIFACTS_DIR.joinpath("metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = None

st.sidebar.header("Options")
mode = st.sidebar.selectbox("Mode", ["Single message", "Batch upload (CSV)"])

# Sidebar: model & parameters
st.sidebar.subheader("Model & parameters")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "SVM"] )
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

# Preprocessing options
st.sidebar.subheader("Text preprocessing")
opt_remove_stopwords = st.sidebar.checkbox("Remove stopwords", value=False)
opt_lowercase = st.sidebar.checkbox("Lowercase", value=True)
opt_remove_punct = st.sidebar.checkbox("Remove punctuation", value=True)

# Display options
st.sidebar.subheader("Display options")
opt_show_preproc = st.sidebar.checkbox("Show preprocessing steps", value=True)
opt_show_probs = st.sidebar.checkbox("Show probability scores", value=True)
opt_show_cm = st.sidebar.checkbox("Show confusion matrix", value=True)
opt_show_roc = st.sidebar.checkbox("Show ROC curve", value=True)

# Retrain button
st.sidebar.markdown("---")
with st.sidebar.expander("Retrain model"):
    retrain_button = st.button("Retrain Model")
    retrain_max_features = st.number_input("Max features (TF-IDF)", min_value=100, max_value=50000, value=5000, step=100)
    retrain_C = st.number_input("Regularization C", min_value=0.0001, max_value=100.0, value=1.0, format="%f")
    retrain_test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.01)

if mode == "Single message":
    text = st.text_area("Enter message text", value="Free entry for demonstration")
    if st.button("Classify"):
        if model is None or vectorizer is None:
            st.error("No model found. Run training first (see README).")
        else:
            # preprocessing function mirrors training options (best-effort)
            def preprocess_text(s: str) -> str:
                if opt_lowercase:
                    s = s.lower()
                if opt_remove_punct:
                    s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
                if opt_remove_stopwords:
                    tokens = [t for t in s.split() if t not in ENGLISH_STOP_WORDS]
                    s = " ".join(tokens)
                return s

            x_raw = text
            x_proc = preprocess_text(x_raw)
            x = [x_proc]
            x_tfidf = vectorizer.transform(x)
            try:
                proba = model.predict_proba(x_tfidf)[0]
                score = float(proba[1])
            except Exception:
                score = None
            label = model.predict(x_tfidf)[0]
            st.metric("Prediction", "SPAM" if label == 1 else "HAM")
            if score is not None and opt_show_probs:
                st.metric("Confidence (spam prob)", f"{score:.3f}")

            # Optionally show whether the pred passes the confidence threshold
            if score is not None:
                st.write(f"Confidence threshold: {confidence_threshold:.2f} — Prediction {'ACCEPTED' if score >= confidence_threshold else 'REJECTED'}")

            # Show preprocessing steps
            st.subheader("Preprocessing Example")
            if opt_show_preproc:
                st.write("Original:", x_raw)
                st.write("Processed:", x_proc)
                tokens = x_proc.split()
                st.write("Tokenized:", tokens)
                if vectorizer is not None:
                    try:
                        feature_names = vectorizer.get_feature_names_out()
                        x_vec = x_tfidf.toarray()[0]
                        top_idx = np.argsort(x_vec)[-10:][::-1]
                        top_features = [(feature_names[i], float(x_vec[i])) for i in top_idx if x_vec[i] > 0]
                        st.write("Top TF-IDF features:")
                        st.table(top_features)
                    except Exception:
                        st.write("TF-IDF feature preview not available")

elif mode == "Batch upload (CSV)":
    uploaded = st.file_uploader("Upload CSV (label,text) or single-column text file", type=["csv", "txt"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, header=None)
            # Heuristic: if two columns assume (label, text), else treat whole file as text column
            if df.shape[1] >= 2:
                texts = df.iloc[:, 1].astype(str).tolist()
            else:
                texts = df.iloc[:, 0].astype(str).tolist()
            if model is None or vectorizer is None:
                st.error("No model available. Train model first.")
            else:
                # apply simple preprocessing to uploaded texts if options set
                def preprocess_list(texts_list):
                    out = []
                    for s in texts_list:
                        if opt_lowercase:
                            s = s.lower()
                        if opt_remove_punct:
                            s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
                        if opt_remove_stopwords:
                            toks = [t for t in s.split() if t not in ENGLISH_STOP_WORDS]
                            s = " ".join(toks)
                        out.append(s)
                    return out

                texts_proc = preprocess_list(texts)
                X_tfidf = vectorizer.transform(texts_proc)
                preds = model.predict(X_tfidf)
                try:
                    probs = model.predict_proba(X_tfidf)[:, 1]
                except Exception:
                    probs = [None] * len(preds)
                out = pd.DataFrame({"text": texts, "processed": texts_proc, "pred": preds, "spam_prob": probs})
                st.dataframe(out)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# Show metrics if available
if metrics is not None:
    st.sidebar.subheader("Model metrics")
    for k, v in metrics.items():
        st.sidebar.write(f"{k}: {v}")

    # Attempt to render confusion matrix and ROC if cached artifacts (placeholders)
    try:
        st.subheader("Evaluation visualizations")
        # Show confusion matrix image
        cm_path = ARTIFACTS_DIR.joinpath("confusion_matrix.png")
        roc_path = ARTIFACTS_DIR.joinpath("roc_curve.png")
        metrics_img = ARTIFACTS_DIR.joinpath("metrics_bar.png")
        if opt_show_cm and cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix")
        elif opt_show_cm:
            st.write("No confusion matrix artifact found. Run training to generate it.")
        if opt_show_roc and roc_path.exists():
            st.image(str(roc_path), caption="ROC curve")
        elif opt_show_roc:
            st.write("No ROC artifact found. Run training to generate it.")
        if metrics_img.exists():
            st.image(str(metrics_img), caption="Primary metrics")
    except Exception:
        st.write("Visualization artifacts not available")

# Handle retrain action
if 'retrain_button' in globals() and retrain_button:
    st.sidebar.info("Retraining model — this may take a minute...")
    # Map model_choice to CLI arg
    cli_model = "lr" if model_choice.lower().startswith("log") else "svm"
    cmd = ["python", "ml/scripts/train_baseline.py", "--model", cli_model, "--max_features", str(int(retrain_max_features)), "--C", str(float(retrain_C)), "--test-size", str(float(retrain_test_size)), "--output-dir", "ml_output"]
    try:
        with st.spinner("Running training script..."):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                st.text(proc.stdout)
            if proc.returncode != 0:
                st.error(f"Training failed (exit {proc.returncode}). See stderr below.")
                if proc.stderr:
                    st.text(proc.stderr)
            else:
                st.success("Retraining completed successfully.")
                # Reload model/metrics
                try:
                    model = joblib.load(MODELS_DIR.joinpath("model.joblib"))
                    vectorizer = joblib.load(MODELS_DIR.joinpath("vectorizer.joblib"))
                    with open(ARTIFACTS_DIR.joinpath("metrics.json"), "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception as e:
                    st.warning(f"Failed to reload artifacts: {e}")
    except Exception as e:
        st.error(f"Failed to run training script: {e}")

st.markdown("---")
st.caption("Demo scaffolding: update `ml/app.py` and training scripts to include richer plots and artifact outputs.")
