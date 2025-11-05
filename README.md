# 2025ML-spamEmail (local workspace)

🌐 **Live Demo**: http://140.120.53.203:8501/  
📦 **GitHub Repository**: https://github.com/g113029016-su/2025ML-spamEmail

This repository holds an OpenSpec-driven project to implement a spam SMS/email classifier and an interactive demo.

Overview
- Phase 1: Baseline classifier using Logistic Regression (optional SVM comparison)
- Phase 1.5: Streamlit demo with rich visualizations and CLI tooling

Repository layout (important files)
- `ml/scripts/train_baseline.py` — CLI training script (uses TF-IDF + LogisticRegression/SVM)
- `ml/app.py` — Streamlit demo app (interactive classification + visualizations)
- `ml/requirements.txt` — Python dependencies
- `openspec/` — specifications and change proposals (use OpenSpec tooling)

Quickstart (local)
1. Create a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml/requirements.txt
```

2. Train baseline model (example):

```powershell
python ml/scripts/train_baseline.py --model lr --max_features 5000 --C 1.0 --output-dir ml_output
```

This will download the dataset, train the model, and save artifacts under `ml_output/`.

3. Run Streamlit demo locally:

```powershell
streamlit run ml/app.py
The app will open at http://localhost:8501
```
Dataset
- Source: Packt tutorial dataset
  - https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Preprocessing: lowercasing, punctuation removal, tokenization via simple whitespace split, and TF-IDF feature extraction (see `ml/scripts/train_baseline.py`).

Notes & Next steps
- Add richer visualizations (training curves, confusion matrix, ROC) to the training script and save plot artifacts to `ml_output/artifacts/` for the Streamlit app to consume.
- Add CI to run `openspec validate` and optional tiny-sample training jobs if desired.

License & attribution
- This project expands code and ideas from the Packt tutorial (see dataset source above). Verify dataset license before using in production.
