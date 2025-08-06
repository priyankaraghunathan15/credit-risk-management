
# Credit Risk Management

## Project Overview
This project predicts the likelihood of loan default using Home Credit's real-world financial data. It integrates manual, automated, and deep learning-based feature engineering techniques with state-of-the-art machine learning models. The project is designed for accurate and reliable credit risk assessment using interpretable and scalable methods.

## Data Sources
- `application_train.csv` / `application_test.csv`: Current loan applications
- Additional client history tables:
  - `bureau`, `bureau_balance`
  - `previous_application`
  - `POS_CASH_balance`
  - `installments_payments`
  - `credit_card_balance`
- Column metadata: `HomeCredit_columns_description.csv`

## Feature Engineering Approaches
- **Manual Aggregation**: Domain-specific engineered features from transactional history.
- **Automated Feature Engineering**: Using Featuretools with a single layer depth on historical tables.
- **Deep Learning-Based Extraction**:
  - CNN-based vector synthesis
  - RNN-based sequential embedding for customer transaction history

## Handling Class Imbalance
- Hierarchical clustering-based undersampling
- SMOTE-based oversampling for minority class balancing
- Combination of both for optimal distribution

## Modeling Techniques
- Gradient Boosting Machines: LightGBM, XGBoost, CatBoost
- Fully Connected Neural Network (FCNN)
- Hyperparameter tuning using Hyperopt (Bayesian optimization)
- Threshold tuning for optimal recall (e.g., threshold = 0.09 for high sensitivity)

## Evaluation Metrics
- Precision, Recall, F1-score
- Area Under ROC Curve (AUC)
- Cohen’s Kappa Score
- Emphasis on Recall to minimize false negatives in loan default prediction

## Notebooks Summary
- `lightgbm_with_cnn_features.ipynb`: CNN feature pipeline with LightGBM classifier
- `lightgbm_with_rnn_features.ipynb`: RNN-based embedding with LightGBM
- `xgboost_with_automated_feature_engineering.ipynb`: Featuretools-generated features with XGBoost

## Folder Structure
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

## How to Run
```bash
# Setup environment
conda env create -f environment_hc.yml
conda activate hc

# Run main script
python main.py --model lightgbm --balance smote
```

## Future Improvements
- Full modularization of codebase
- CLI-driven model training options
- Unit testing and continuous integration (CI) for code reliability

---
