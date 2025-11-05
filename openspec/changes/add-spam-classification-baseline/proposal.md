# Change: Add spam email classification baseline (phase1-baseline)

## Why
We want a baseline machine-learning solution to classify spam SMS/email messages so the project can iterate on improving detection quality and integrating ML-driven protections. A small, well-documented baseline makes it easier to measure progress across phases and to validate further improvements.

## What Changes
- Add a baseline ML pipeline that trains and evaluates a spam classifier using a public SMS spam dataset.
- Phase 1 (baseline): implement a simple, reproducible training pipeline and evaluation using logistic regression (and optionally SVM for comparison). The initial runnable artifact will be a script/notebook that downloads the CSV, trains the model, outputs evaluation metrics, and saves the trained model and artifacts.
- Future phases (placeholders): extend to larger datasets, feature engineering, hyperparameter tuning, model deployment, monitoring. These will be added as separate phases (phase2, phase3, ...).

### Tooling / CI
- Add a CI check that runs `openspec validate <change-id> --strict` for proposed changes under `openspec/changes/` to ensure spec correctness before merge.
- Add a sample GitHub Actions workflow that installs Node and `@fission-ai/openspec` (or uses `npx`) and runs `openspec validate` for changed `openspec/changes/*` directories.

## Data Source
- Training dataset (CSV): https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Note: this CSV appears to be prepackaged and public; verify licensing and suitability before using in production.

## Phase 1 (phase1-baseline)
Goals
- Produce a minimal, reproducible baseline classifier with clearly documented preprocessing, training, and evaluation steps.
- Provide scripts and a notebook demonstrating how to run training and evaluation locally.

Deliverables
- `scripts/train_baseline.py` or `notebooks/train_baseline.ipynb` that:
  - Downloads and reads the CSV dataset
  - Performs minimal preprocessing (tokenization, lowercasing, removing punctuation, simple TF-IDF or count vectorizer)
  - Trains a Logistic Regression classifier (primary) and optionally trains an SVM for comparison
  - Evaluates with accuracy, precision, recall, F1, and ROC-AUC
  - Saves the model and a small report (JSON/CSV) with metrics
- A short README snippet describing how to run the baseline and reproduce results

Assumptions
- We'll assume a Python-based ML toolchain (Python 3.8+, scikit-learn, pandas). If you'd prefer a Node.js-based ML approach, say so and I will adjust the plan.
- The user mentioned "logistic regression" and also "SVM"; phase1 will implement Logistic Regression as the primary baseline and include optional SVM training for side-by-side comparison.

## Phase 1.5 (streamlit-demo + visualizations)
Purpose
- Add an interactive demo and rich visualizations to make the baseline accessible to non-developers and to aid analysis of preprocessing and model behavior. This phase is intentionally separate so reviewers can approve the baseline training artifacts independently before the user-facing demo is added.

Deliverables (Phase 1.5)
- Streamlit demo app (`app.py`) that provides an interactive web interface where users can:
  - Input a single message text or upload a small text file / CSV for batch prediction
  - Receive classification results (spam/ham) and show confidence scores for each prediction
  - View visualizations for preprocessing steps and model metrics
  - Be deployable to Streamlit Community Cloud at `https://2025spamemail.streamlit.app/` (deployment instructions included)
- Rich visualizations:
  - Output examples for each preprocessing step (original text → tokenized → cleaned → TF-IDF vector snippets)
  - Training metrics charts (accuracy over training steps, loss curves if applicable)
  - Confusion matrix and ROC curve visualization
  - Model comparison charts (Logistic Regression vs SVM performance across metrics)
  - A step-by-step pipeline visualization (data → preprocess → features → model → eval)
- CLI tool (optional but recommended):
  - `scripts/train_baseline.py` should accept CLI args for model selection and hyperparameters (example: `python train_baseline.py --model lr --max_features 5000 --C 1.0`)
  - Provide `scripts/predict.py` or `app.py` integration for CLI prediction
- Documentation and repo integration:
  - `README.md` with project structure, how to run locally, how to deploy to Streamlit, dataset source and preprocessing steps
  - `requirements.txt` listing Python dependencies (scikit-learn, pandas, streamlit, matplotlib/seaborn, joblib, etc.)
  - Instructions / CI steps to push to `https://github.com/g113029016-su/2025ML-spamEmail`

Notes
- Streamlit deployment: provide a short deployment section in the README and a sample `Procfile` or instructions for Streamlit Community Cloud. Ensure secrets and large artifacts (models) are not committed; use artifacts or model hosting if needed.

## Impact
- Affected files: new scripts/notebooks under a `ml/` or `scripts/` folder, `app.py` (Streamlit), `requirements.txt`, `README.md`, and documentation updates. No changes to production services are necessary for phase1/1.5 other than CI tooling.
- CI: add test(s to confirm the training script runs on a small sample (optional) and add validation for `openspec` deltas as previously described.

## Impact: Tooling
- Affected tooling: CI configuration (e.g., `.github/workflows/openspec-validate.yml`) and developer docs (`openspec/project.md`) to recommend running `openspec validate` on PRs.
- Developer workflow change: PRs that alter `openspec/changes/` will run the validation check; contributors may run `openspec validate` locally before opening PRs.

## Risks & Mitigations
- Data licensing: verify dataset license before reuse in production.
- Environment differences: provide a `requirements.txt` and/or `environment.yml` for reproducibility.
- Large datasets: initial dataset is small; keep pipeline memory-light and deterministic.

## Rollout Plan
1. Add this proposal and tasks.
2. Implement `phase1-baseline` scripts/notebook and add `requirements.txt`.
3. Run local training, evaluate, and check-in artifacts and small test ensuring training runs on CI if desired.
4. Iterate based on evaluation and add phase2 proposals for improvements.

***

If this looks good I will scaffold `tasks.md` and a spec delta under `openspec/changes/add-spam-classification-baseline/specs/ml/spec.md` and commit the files. If you prefer SVM instead of logistic regression for the primary baseline, tell me and I'll update the proposal accordingly.
