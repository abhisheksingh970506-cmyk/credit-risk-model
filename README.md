# 🏦 Credit Risk Modelling — Predicting Loan Default with Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange) ![AUC](https://img.shields.io/badge/AUC-0.8418-brightgreen) ![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Project Overview

This project builds a **binary credit risk classification model** to predict whether a borrower will experience serious financial distress (90+ days past due) within the next two years. The goal is to support lenders in making smarter, data-driven credit decisions by identifying high-risk applicants before loan approval.

**Dataset:** Kaggle — [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)  
**Model:** Logistic Regression  
**Final AUC:** 0.8418 (Stratified Test) | 0.8426 (Out-of-Time Validation)

---

## 🎯 Business Problem

Banks and financial institutions face significant losses from loan defaults. Traditional credit scoring methods can miss subtle risk signals hidden in financial behaviour data. This model provides:

- A **probability of default (PD)** for each borrower
- A **credit score** derived from predicted probabilities
- An **expected loss estimate** to support portfolio risk management
- A **threshold analysis** to balance approval rates against risk exposure

---

## 📂 Project Structure

```
credit-risk-model/
│
├── Credit_Modelling.ipynb       # Main notebook (EDA, modelling, evaluation)
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── outputs/
    └── submission.csv           # Final predictions on test set
```

---

## 🔄 Methodology

### 1. Data Loading & Inspection
- Dataset: 150,000 borrower records with 10 financial features
- Target variable: `SeriousDlqin2yrs` (1 = default, 0 = no default)
- Class imbalance identified: ~6.7% default rate

### 2. Data Cleaning
| Issue | Action Taken |
|---|---|
| MonthlyIncome — 19.82% missing | Imputed with median |
| NumberOfDependents — 2.62% missing | Imputed with mode |
| Age < 18 (invalid entries) | Dropped (2.2% of data) |
| RevolvingUtilization > 1 (data errors) | Dropped alongside age filter |
| Outliers across numeric columns | Clipped at 1st–99th percentile |

**Cleaned dataset: 146,678 rows**

### 3. Exploratory Data Analysis (EDA)
- Distribution plots for all features
- Correlation heatmap — delinquency and utilisation strongly correlated with default
- Default rate by age bin — younger borrowers carry higher risk
- Weight of Evidence (WOE) binning for key variables

### 4. Feature Engineering
| New Feature | Logic |
|---|---|
| `TotalDelinquencies` | Sum of all 3 past-due count columns |
| `RealEstateToCredit` | Real estate loans / total open credit lines |
| `MonthlyIncome_log` | Log-transform to reduce right skew |
| `DebtRatio_log` | Log-transform to reduce right skew |

**Total features used in model: 14**

### 5. Model Training
- Algorithm: **Logistic Regression** (`liblinear` solver, L2 regularisation)
- Train/test split: **80/20 stratified** (random_state=42)
- Stratification ensures class balance is preserved in both splits

### 6. Evaluation

| Metric | Value |
|---|---|
| **AUC Score (Stratified Test)** | **0.8418** |
| **AUC Score (Out-of-Time Test)** | **0.8426** |
| KS Statistic | 0.5276 |
| Average Precision | 0.3677 |
| Optimal Decision Threshold | 0.0663 |
| Recall at Threshold | 0.76 |
| Population Stability Index (PSI) | 0.0002 ✅ Very Stable |

### 7. Business Output
- **Expected Loss** calculated per borrower using predicted PD
- **Credit Score Scale** derived from model probabilities
- **Threshold Analysis** — visualises trade-off between approval rate and expected loss exposure
- **Calibration Curve** — confirms predicted probabilities are reliable estimates of actual default likelihood
- **Decile Table** — model scores ranked into 10 buckets for portfolio segmentation

---

## 📈 Key Results

- The model achieved an **AUC of 0.8418**, indicating strong discriminatory power between defaulters and non-defaulters
- **KS Statistic of 0.5276** — excellent separation between good and bad score distributions
- **PSI of 0.0002** — near-zero score drift between train and test sets, confirming model stability
- Out-of-Time validation AUC of **0.8426** — model generalises well to unseen, chronologically later data
- Class imbalance (6.7% default rate) handled via stratified sampling; precision-recall trade-off clearly documented

---

## ⚙️ How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook Credit_Modelling.ipynb
```

> **Note:** Place `cs-training.csv` and `cs-test.csv` in the same directory (or update the file paths in the notebook to match your local setup). Dataset available on [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data).

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🔮 Future Improvements

- **Gradient Boosting models** (XGBoost, LightGBM) to capture non-linear patterns
- **SMOTE / class weighting** to improve precision on the minority class
- **Hyperparameter tuning** via GridSearchCV for optimised regularisation
- **Interaction features** (e.g., age × monthly income) for richer signal
- **Automated monitoring pipeline** tracking AUC, KS, and PSI in production

---

## 👤 Author

**Abhishek Singh**  
MBA Finance | Computer Science Engineering  
[GitHub](https://github.com/abhisheksingh970506-cmyk) | [LinkedIn](https://linkedin.com/in/)

---

## 📄 License

This project is for educational and portfolio purposes. Dataset credit: Kaggle — Give Me Some Credit Competition.
