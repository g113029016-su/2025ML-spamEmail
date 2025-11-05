# 2025ML-spamEmail (local workspace)

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
```

Deploying to Streamlit Community Cloud
- Create a GitHub repository: https://github.com/huanchen1107/2025ML-spamEmail
- Push your code (do NOT push large model artifacts). Streamlit will install `ml/requirements.txt` and run `streamlit run ml/app.py`.
- Follow Streamlit Community Cloud instructions to configure repo and deploy to `https://2025spamemail.streamlit.app/`.

Dataset
- Source: Packt tutorial dataset
  - https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Preprocessing: lowercasing, punctuation removal, tokenization via simple whitespace split, and TF-IDF feature extraction (see `ml/scripts/train_baseline.py`).

Notes & Next steps
- Add richer visualizations (training curves, confusion matrix, ROC) to the training script and save plot artifacts to `ml_output/artifacts/` for the Streamlit app to consume.
- Add CI to run `openspec validate` and optional tiny-sample training jobs if desired.

License & attribution
- This project expands code and ideas from the Packt tutorial (see dataset source above). Verify dataset license before using in production.

## Deploying to GitHub and Streamlit Cloud

This project can be pushed to GitHub and deployed to Streamlit Community Cloud.

1) Push to GitHub (example using the repo you provided):

```powershell
git init
git branch -M main
git remote add origin https://github.com/huanchen1107/2025ML-spamEmail.git
git add -A
git commit -m "Initial commit: spam classifier demo and OpenSpec proposals"
git push -u origin main
```

Notes:
- If your local git is not configured with user.name/user.email, set them before committing.
- If push prompts for credentials, either use HTTPS credentials or configure an SSH key and push with `git@github.com:huanchen1107/2025ML-spamEmail.git`.

2) Deploy to Streamlit Community Cloud:

- Create a GitHub repo (or use the repo above) and push the code.
- On https://streamlit.io/cloud, choose "New app" and connect your GitHub repo, pick the `main` branch and the `ml/app.py` file as the main program.
- Streamlit will install `ml/requirements.txt`. If you need large models (>100MB), do NOT commit them — instead have your app train on first run or download a model from a hosted artifact store.
- After deployment completes, Streamlit will provide a shareable URL (like `https://share.streamlit.io/<user>/<repo>/<branch>/ml/app.py` or a custom domain). If you want, I can guide you step-by-step through the Streamlit UI flow.

3) Large model handling

- Avoid checking in models >100MB. Two approaches:
  - Train during deployment: have the app or an initial CI step run `python ml/scripts/train_baseline.py --output-dir ml_output` to produce artifacts at first run.
  - Host model artifacts on a storage service (S3, GitHub Releases) and download at startup.

If you want me to push the repo now and try to deploy, I can attempt to push (you may be prompted for credentials in this environment). If push is successful, I can walk you through the Streamlit Cloud steps or attempt to create the app if you give me the Streamlit account access (not recommended to share secrets here).
