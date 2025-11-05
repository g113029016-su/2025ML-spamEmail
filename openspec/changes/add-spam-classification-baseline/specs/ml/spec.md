## ADDED Requirements

### Requirement: Baseline spam classification (phase1-baseline)
The project SHALL provide a reproducible baseline machine learning pipeline that trains and evaluates a spam message classifier using the public SMS spam dataset.

#### Scenario: Dataset ingestion
- **WHEN** the training pipeline runs
- **THEN** it SHALL download and read the CSV dataset from `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv`
- **THEN** it SHALL parse the dataset into features (message text) and labels (spam/ham)

#### Scenario: Training baseline model
- **WHEN** preprocessing completes
- **THEN** the pipeline SHALL train a Logistic Regression classifier using TF-IDF or count vector features
- **THEN** the pipeline SHALL produce evaluation metrics including accuracy, precision, recall, F1, and ROC-AUC
- **THEN** the pipeline SHALL save model artifacts (model file and a metrics report)

#### Scenario: Local reproducibility
- **WHEN** a developer runs the baseline training script with documented instructions and dependencies
- **THEN** the run SHALL be reproducible given the same random seed and environment

### Requirement: Streamlit demo application (phase1.5)
The project SHALL provide a Streamlit-based demo application that allows interactive classification and visualization of model behavior.

#### Scenario: Single message prediction
- **WHEN** a user opens the Streamlit app and enters message text
- **THEN** the app SHALL return a classification (spam/ham) and a confidence score for the prediction

#### Scenario: Batch file prediction
- **WHEN** a user uploads a small CSV or text file to the app
- **THEN** the app SHALL process the file and return a list of predictions with confidence scores

#### Scenario: Deployment
- **WHEN** maintainers follow the provided deployment instructions
- **THEN** the app SHALL be deployable to Streamlit Community Cloud at `https://2025spamemail.streamlit.app/`

### Requirement: Rich visualizations (phase1.5)
The project SHALL include visualization outputs demonstrating preprocessing, training metrics, and model comparisons.

#### Scenario: Preprocessing examples
- **WHEN** the user requests to view preprocessing examples in the app or notebook
- **THEN** the system SHALL display example transformations: original text → tokenized → cleaned tokens → TF-IDF vector snippets

#### Scenario: Training metrics and evaluation
- **WHEN** training completes or metrics are available
- **THEN** the system SHALL render charts for accuracy curves, (loss curves if applicable), confusion matrix, and ROC curve

#### Scenario: Model comparison
- **WHEN** both Logistic Regression and SVM models are trained
- **THEN** the system SHALL display side-by-side comparison charts for accuracy, precision, recall, F1, and ROC-AUC

### Requirement: CLI tool (phase1.5)
The project SHOULD provide command-line interfaces for training and prediction with adjustable hyperparameters.

#### Scenario: Command-line training
- **WHEN** a developer runs `python scripts/train_baseline.py --model lr --max_features 5000 --C 1.0`
- **THEN** the script SHALL train the selected model with the provided hyperparameters and output metrics and saved artifacts

#### Scenario: Command-line prediction
- **WHEN** a developer runs `python scripts/predict.py --model-path model.joblib --text "Example spam message"`
- **THEN** the script SHALL print the predicted label and confidence score to stdout and return a zero exit code on success

### Requirement: GitHub integration & docs (phase1.5)
The repository SHALL include documentation and CI-ready artifacts so the project can be pushed to `https://github.com/huanchen1107/2025ML-spamEmail` and run by contributors.

#### Scenario: README and requirements
- **WHEN** contributors view the repository
- **THEN** they SHALL find a `README.md` describing project structure, how to run locally, how to deploy to Streamlit, dataset source, and preprocessing steps
- **THEN** they SHALL find `requirements.txt` listing required Python packages

#### Scenario: Repo push
- **WHEN** maintainers push the code and docs to the GitHub repo
- **THEN** everything necessary to run the demo locally (except large model artifacts) SHALL be present and documented

### Requirement: Project Context Attribution
The project SHALL reference the Packt tutorial as the initial source material and declare that this work expands Chapter 3's spam detection with enhanced preprocessing, an interactive demo, and richer visualizations.


