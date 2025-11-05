## 1. Implementation (phase1-baseline)
- [ ] 1.1 Create `ml/` or `scripts/` folder for baseline artifacts
- [ ] 1.2 Add `requirements.txt` with minimal dependencies (pandas, scikit-learn, joblib)
- [ ] 1.3 Implement `scripts/train_baseline.py` (data download, preprocess, train, eval, save)
- [ ] 1.4 Add `notebooks/train_baseline.ipynb` demonstrating steps (optional but recommended)
- [ ] 1.5 Add a reproducible random seed and a small sample unit test that runs the script on a subset
- [ ] 1.6 Document how to run locally in README or `ml/README.md`
- [ ] 1.7 Add CI workflow to run `openspec validate` on PRs that change `openspec/changes/`
	- [ ] 1.7.1 Create `.github/workflows/openspec-validate.yml` that installs Node and runs `npx @fission-ai/openspec validate <change-id> --strict` for changed changes
	- [ ] 1.7.2 Add a developer npm script (e.g., `npm run openspec-validate`) to run validation locally
	- [ ] 1.7.3 Document CI requirement in `openspec/project.md`

## 1.5 Implementation (streamlit-demo + visualizations)
- [ ] 1.5.1 Create `app.py` (Streamlit) that:
	- Accepts single-message input and small-file/CSV upload for batch prediction
	- Calls the trained model to produce predictions and confidence scores
	- Renders visualizations for preprocessing and evaluation metrics
- [ ] 1.5.2 Implement visualization helpers:
	- Output examples for each preprocessing step (tokenization, cleaned tokens, TF-IDF snippets)
	- Plot training metrics (accuracy, loss if available) and ROC curve
	- Plot confusion matrix and model comparison charts (LR vs SVM)
- [ ] 1.5.3 Add CLI argument parsing to `scripts/train_baseline.py` and provide `scripts/predict.py` for command-line predictions
- [ ] 1.5.4 Add `requirements.txt` entries for Streamlit and plotting libraries (streamlit, matplotlib, seaborn, plotly optional)
- [ ] 1.5.5 Add `README.md` sections describing how to run the Streamlit app locally and how to deploy to Streamlit Community Cloud (`https://2025spamemail.streamlit.app/`)
- [ ] 1.5.6 Provide a sample deployment instruction and guardrails to avoid committing large model artifacts or secrets

## 2. Verification (updated)
- [ ] 2.1 Run training locally and record baseline metrics
- [ ] 2.2 Ensure script exits with meaningful error codes on failure
- [ ] 2.3 Verify the Streamlit app runs locally and returns predictions for a sample input
- [ ] 2.4 Confirm visualizations render and metrics match training output
- [ ] 2.5 (Optional) Add CI job to run baseline on a tiny sample

## 2. Verification
- [ ] 2.1 Run training locally and record baseline metrics
- [ ] 2.2 Ensure script exits with meaningful error codes on failure
- [ ] 2.3 (Optional) Add CI job to run baseline on a tiny sample

## 3. Future Phases (placeholders)
- [ ] phase2: (placeholder) feature engineering and hyperparameter search
- [ ] phase3: (placeholder) model deployment / API wrapping
- [ ] phase4: (placeholder) monitoring and retraining workflow
