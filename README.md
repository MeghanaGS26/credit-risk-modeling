# Credit Risk Modeling

Predicting which loan applicants are likely to default, with explainable machine learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeghanaGS26/credit-risk-modeling/blob/main/Credit_Risk_Modeling.ipynb)

## Problem Statement
Lenders need to assess the creditworthiness of applicants. This project predicts the probability of serious delinquency within two years and explains what drives it.

## Dataset
"Give Me Some Credit" (Kaggle competition, `cs-training.csv`): 150,000 borrowers, 10 features, target `SeriousDlqin2yrs`. Only 6.7% of borrowers defaulted. The data is not included here, so download it from the Kaggle competition page and upload it to the notebook.

## Approach
1. Cleaning: removed an impossible age (0), converted placeholder codes (96, 98) in late-payment columns to missing
2. EDA: default rate by age, late payments and credit utilization
3. Stratified 80/20 train-test split, then median imputation and 99th-percentile capping fitted on training data only (to avoid leakage)
4. Imbalance handling: SMOTE applied to the training set only
5. Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
6. Evaluation on the untouched test set: accuracy, precision, recall, F1, ROC-AUC, plus decision-threshold tuning
7. Explainability: SHAP

## Results (threshold 0.5)
| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.818 | 0.230 | 0.737 | 0.861 |
| Decision Tree | 0.922 | 0.400 | 0.324 | 0.825 |
| Random Forest | 0.927 | 0.441 | 0.353 | 0.860 |
| XGBoost | 0.934 | 0.523 | 0.222 | 0.862 |

Accuracy is misleading on this data: predicting "no default" for everyone already scores about 93%.

## Threshold tuning (XGBoost)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.1 | 0.208 | 0.789 | 0.329 |
| 0.2 | 0.329 | 0.580 | 0.420 |
| 0.3 | 0.407 | 0.439 | 0.422 |
| 0.5 | 0.523 | 0.222 | 0.312 |

At threshold 0.2, the model catches 1,163 of 2,005 defaulters in the test set, at the cost of 2,372 good applicants being flagged for review.

## Key Insights
- Payment history (30-59, 60-89 and 90+ day late payments) is the strongest driver of default risk.
- High credit utilization increases risk.
- Older borrowers and higher-income borrowers show lower risk.

## Business Recommendations
1. Prioritize payment history and credit utilization in screening.
2. Use a lower threshold (0.2) and route flagged applicants to manual review.
3. Use SHAP explanations to justify decisions transparently.

## Tools Used
Python, pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, SHAP, matplotlib, seaborn, Google Colab

## How to Run
Download `cs-training.csv` from the Kaggle competition, open the notebook in Colab, upload the file, and choose **Runtime → Run all**.
