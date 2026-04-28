# Lending Club Credit Risk Research

Machine learning project for predicting default risk in LendingClub-style peer-to-peer lending data.

This repository develops, compares, and evaluates credit risk models using Logistic Regression, Random Forest, and XGBoost. The project focuses not only on predictive performance, but also on common real-world machine learning issues in finance, including data leakage, preprocessing, metric selection, class imbalance, threshold choice, and business decision-making.

## Project summary

The aim of this project is to predict whether a loan applicant is likely to default and to translate model predictions into a practical lending decision rule. The workflow evaluates multiple models on a held-out test set and compares them using AUC-ROC, precision, recall, F1 score, training time, and expected profit under a simplified lending payoff framework.

The final report recommends an Optuna-tuned XGBoost model as the strongest production candidate, while retaining Logistic Regression as a transparent benchmark.

## Repository structure

```text
.
├── LendingClub.csv
├── credit_risk_research_notebook.ipynb
├── credit_risk_research_report.pdf
├── .gitattributes
└── README.md
```

## Files

| File | Description |
|---|---|
| `LendingClub.csv` | LendingClub dataset used for model training and evaluation. Stored with Git LFS because it is too large for normal Git tracking. |
| `credit_risk_research_notebook.ipynb` | Main Jupyter notebook containing preprocessing, model training, tuning, evaluation, feature importance, and threshold analysis. |
| `credit_risk_research_report.pdf` | Final written report summarising methodology, debugging fixes, results, business recommendations, and limitations. |
| `.gitattributes` | Git LFS tracking configuration for CSV files. |

## Methodology

The modelling workflow includes:

1. Loading and inspecting the LendingClub dataset.
2. Cleaning and preprocessing mixed numerical and categorical features.
3. Treating special missing-value encodings such as `999` in `mths_since_last_delinq`.
4. Creating a stratified train-test split.
5. Building separate preprocessing pipelines for linear and tree-based models.
6. Training and tuning:
   - Logistic Regression
   - Random Forest
   - XGBoost
7. Comparing models using:
   - AUC-ROC
   - Precision
   - Recall
   - F1 score
   - Training time
8. Analysing feature importance.
9. Testing classification thresholds using an expected profit rule.
10. Producing business recommendations and next steps.

## Models

### Logistic Regression

Used as the interpretable benchmark model. Numerical features are median-imputed and standardised, while categorical features are imputed and one-hot encoded. Class imbalance is handled using class weighting.

### Random Forest

Used to capture non-linear relationships and feature interactions. The model is tuned using cross-validation and class weighting.

### XGBoost

Used as the main high-performance gradient boosting model. Class imbalance is addressed using `scale_pos_weight`, and hyperparameters are tuned using both GridSearchCV and Optuna.

## Key results

The final report compares GridSearchCV and Optuna-tuned versions of each model. The strongest model was the Optuna-tuned XGBoost model.

| Model | Tuning method | AUC-ROC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | GridSearchCV | 0.7071 | 0.3131 | 0.6692 | 0.4266 |
| Logistic Regression | Optuna | 0.7071 | 0.3131 | 0.6691 | 0.4266 |
| Random Forest | GridSearchCV | 0.7114 | 0.3197 | 0.6536 | 0.4293 |
| Random Forest | Optuna | 0.7114 | 0.3185 | 0.6599 | 0.4296 |
| XGBoost | GridSearchCV | 0.7184 | 0.3208 | 0.6760 | 0.4351 |
| XGBoost | Optuna | 0.7195 | 0.3217 | 0.6747 | 0.4357 |

## Threshold and profit analysis

The project does not assume that a default classification threshold of `0.50` is automatically optimal. Instead, thresholds are evaluated using the simplified expected profit formula from the brief:

```text
Expected Profit = (TN × $2,250) - (FN × $15,000)
```

The best tested threshold was `0.40`, which generated the highest expected profit under the project assumptions.

| Threshold | Expected profit |
|---:|---:|
| 0.30 | $105,157,500 |
| 0.40 | $116,572,500 |
| 0.50 | $66,507,750 |
| 0.60 | -$58,070,250 |
| 0.70 | -$224,643,750 |

## Main fixes made to the starter workflow

The starter workflow contained four common machine learning issues:

### 1. Data leakage and look-ahead bias

Hyperparameter tuning was originally exposed to the test set. This was corrected by using cross-validation on the training set only and reserving the test set for final evaluation.

### 2. Incomplete feature preprocessing

The corrected workflow uses `Pipeline` and `ColumnTransformer` objects so that numerical and categorical preprocessing are handled properly and fitted only on training folds.

### 3. Weak metric selection

Accuracy-style evaluation was replaced with AUC-ROC for cross-validation, plus precision, recall, and F1 score for test-set classification performance.

### 4. Arbitrary business threshold

The fixed `0.50` threshold was replaced with threshold testing based on expected profit, making the final decision rule more aligned with lending economics.

## Installation

Clone the repository:

```bash
git clone https://github.com/lhkkennedy/Lending-Club-Credit-Risk-Research.git
cd Lending-Club-Credit-Risk-Research
```

Install Git LFS before pulling the dataset:

```bash
git lfs install
git lfs pull
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn optuna xgboost jupyter
```

## Running the notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
credit_risk_research_notebook.ipynb
```

If the notebook expects a different dataset filename, update the `pd.read_csv(...)` line so it matches the local CSV file name, for example:

```python
df = pd.read_csv("LendingClub.csv")
```

## Git LFS note

The dataset is larger than GitHub's standard file-size limit, so CSV files are tracked using Git LFS.

To check LFS tracking:

```bash
git lfs track
git lfs status
```

To add a large CSV correctly:

```bash
git lfs track "*.csv"
git add .gitattributes
git add LendingClub.csv
git commit -m "Add LendingClub dataset with Git LFS"
```

## Limitations

This project should be interpreted as a research and coursework-style credit risk modelling exercise rather than a production-ready lending system. Important limitations include:

- The dataset is historical and may reflect past underwriting policy.
- The simplified profit model excludes recoveries, funding costs, servicing costs, prepayment risk, and opportunity costs.
- LendingClub `grade` and `interest_rate` may encode information from the platform's existing underwriting process.
- Fairness testing is required because borrower and loan variables may proxy for protected characteristics.
- XGBoost requires explainability checks and model monitoring before deployment.

## Suggested next steps

Future work could include:

- Out-of-time validation using later loan cohorts.
- Probability calibration checks.
- Fairness and bias testing.
- SHAP-based explainability for XGBoost.
- Testing a model that excludes `grade`, `interest_rate`, and other potentially underwriting-derived variables.
- Monitoring model drift and threshold performance over time.
- Comparing approval-rate, default-rate, and profit trade-offs under different risk appetites.

## Author

Lewis Hikari Kawase Kennedy

MSc Finance and Machine Learning  
Queen Mary University of London
