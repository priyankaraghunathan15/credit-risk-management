
# Credit Risk Management Using Machine Learning

This project builds a scalable machine learning pipeline to predict the risk of loan default using the Home Credit dataset. By integrating manual, automated, and deep learning-based feature engineering with traditional and deep learning models, it offers a robust framework for creditworthiness assessment in financial services.

---

## Project Objectives

- Predict the probability of loan repayment default
- Apply multiple feature engineering strategies (manual, automated, and deep learning-based)
- Address class imbalance using hybrid sampling techniques
- Evaluate performance using metrics tailored to high-risk financial decision-making

---

## Data Sources

- `application_train.csv`, `application_test.csv`: Current loan application data
- Supplementary historical and transactional data:
  - `bureau`, `bureau_balance`
  - `previous_application`
  - `POS_CASH_balance`
  - `installments_payments`
  - `credit_card_balance`
- Metadata: `HomeCredit_columns_description.csv`

---

## Feature Engineering Methods

- Manual Aggregation: Domain-specific feature creation from historical tables using group statistics (mean, sum, count).
- Automated Feature Engineering: One-depth relational features using Featuretools.
- Deep Learning-Based Extraction:
  - CNN-based representations to learn feature patterns
  - RNN embeddings to model time-dependent customer behavior

---

## Class Imbalance Strategy

- Hierarchical clustering-based undersampling of majority class
- SMOTE oversampling for minority class
- Combined sampling strategies to improve generalization

---

## Models and Optimization

- Models Used: LightGBM, XGBoost, CatBoost, Fully Connected Neural Network (FCNN)
- Tuning: Bayesian optimization using Hyperopt
- Categorical Encoding:
  - Category dtype for LGBM & CatBoost
  - One-hot encoding for XGBoost & FCNN
- Threshold Optimization: Custom classification thresholds (e.g., 0.09) to optimize recall

---

## Evaluation Metrics

| Metric           | Purpose                                                  |
|------------------|----------------------------------------------------------|
| Recall           | Prioritizes detection of true defaulters                 |
| Precision        | Ensures flagged loans are truly risky                    |
| F1-Score         | Balances precision and recall                            |
| AUC-ROC          | Assesses overall classifier performance                  |
| Cohen’s Kappa    | Evaluates agreement between prediction and ground truth |

---

## Sample Results

| Model         | Recall | Precision | F1-Score | AUC   | Cohen’s Kappa |
|---------------|--------|-----------|----------|-------|----------------|
| LightGBM      | 0.82   | 0.41      | 0.55     | 0.77  | 0.50           |
| XGBoost       | 0.80   | 0.44      | 0.57     | 0.78  | 0.53           |
| CatBoost      | 0.81   | 0.43      | 0.56     | 0.79  | 0.51           |
| FCNN          | 0.75   | 0.38      | 0.50     | 0.74  | 0.47           |

---

## Key Insights

- LightGBM and XGBoost models outperformed other classifiers, achieving strong recall scores while maintaining moderate precision.
- Automated feature engineering via Featuretools yielded competitive results, reducing manual overhead while maintaining model quality.
- CNN and RNN-based embeddings provided incremental improvements, though tree-based models remained more interpretable and efficient.
- Threshold tuning significantly improved recall, which is critical in credit risk contexts where failing to detect a defaulter is costlier than false positives.
- Combining undersampling and SMOTE led to a more balanced dataset, allowing models to generalize better across minority and majority classes.

---

## Notebook Summaries

| Notebook Filename                              | Description |
|------------------------------------------------|-------------|
| `lightgbm_with_cnn_features.ipynb`             | Extracts high-level features using CNNs before training a LightGBM classifier |
| `lightgbm_with_rnn_features.ipynb`             | Captures temporal dependencies with RNN-based embeddings and feeds into LightGBM |
| `xgboost_with_automated_feature_engineering.ipynb` | Uses Featuretools to auto-generate features from multi-table input, followed by XGBoost training |

---

## Repository Structure

```
Credit-Risk-Management/
├── data/                    # CSVs and raw data files
├── notebooks/               # EDA and modeling notebooks
├── src/                     # Python scripts (main.py, model.py)
├── results/                 # Model outputs and reports
├── images/                  # Figures and diagrams
├── environment_hc.yml       # Conda environment file
└── README.md
```

---

## Entity Relationship Diagram

![Entity Relationship Diagram](images/credit-risk-diagram.png)

---

## How to Run

```bash
# Create and activate environment
conda env create -f environment_hc.yml
conda activate hc

# Run pipeline (choose model: lightgbm, xgboost, catboost, fcnn)
python main.py --model lightgbm --balance smote --threshold 0.09
```

---

## Next Steps

- Modularize codebase into reusable pipeline components
- Add command-line argument parsing for flexibility
- Integrate unit testing and CI/CD for reliability
- Build interactive dashboard or reporting layer for live insights

---

## Conclusion

This project demonstrates the integration of advanced modeling and feature engineering techniques to build a reliable credit risk prediction system. By combining classical ML methods with deep learning-inspired feature extraction and robust handling of imbalanced data, the pipeline balances performance with interpretability—an essential trade-off in financial risk modeling. The results are promising and adaptable for real-world deployment in credit evaluation workflows.

---
