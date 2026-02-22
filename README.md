# Health Insurance Claim Predictor

Predicts annual insurance claim amounts from policyholder health and demographic data. Built as a complete ML pipeline — from raw CSV to a deployed Streamlit app.

**Live app →** https://health-insurance-claim-prediction-1-5bf4.onrender.com

---

## Executive Summary

This project builds an end-to-end machine learning system that estimates annual medical insurance claim costs.

The system includes:

- Data audit and exploratory analysis  
- Feature engineering grounded in healthcare risk logic  
- Baseline benchmarking  
- Tree-based model comparison  
- Hyperparameter tuning using GridSearchCV  
- Cross-validation for generalization  
- Residual diagnostics  
- Deployment as a live Streamlit web application  

The final XGBoost model achieved:

- **Test R²: 0.81**
- **MAE: ~$3,800**
- **RMSE: $5,263**
- Significant improvement over a mean-baseline model

This project demonstrates production-ready ML system design, not just model training.

---

## Business Problem

Insurance companies rely on claim cost estimation to:

- Price premiums accurately  
- Manage financial risk  
- Identify high-risk policyholders  
- Reduce underwriting uncertainty  

Poor estimation can result in financial losses for insurers or unfair pricing for customers.

This project builds a regression model that predicts expected claim amounts based on measurable health and demographic attributes.

Dataset size: 1,340 records (1,332 after cleaning).

Target variable: Annual insurance claim amount (USD).

---

## Key Insights from Analysis

Exploratory data analysis revealed:

- **Smoking status is the strongest predictor of claim cost.**
- Smokers claim significantly more than non-smokers.
- Blood pressure shows stronger correlation with claim amount than age.
- High BMI combined with smoking amplifies risk.
- Regional variation affects cost, likely reflecting underlying healthcare pricing differences.

These insights informed model selection and evaluation.

---

## Methodology

### Data Preparation

- Removed <1% missing values to avoid introducing imputation bias.
- Performed univariate and bivariate analysis.
- Applied log transformation to the target to reduce right skew.
- Split data into train/test sets (80/20).

### Modeling Pipeline

A full Scikit-learn `Pipeline` with `ColumnTransformer` was implemented:

- Numerical features → StandardScaler  
- Categorical features → OneHotEncoder  
- Model → XGBoost  

This ensures:

- No data leakage  
- Consistent preprocessing  
- Clean deployment integration  

### Models Compared

| Model | Test R² | MAE | RMSE |
|--------|--------|--------|--------|
| Baseline (Mean Predictor) | -0.13 | $8,194 | — |
| Random Forest | 0.80 | $3,820 | $5,316 |
| **XGBoost** | **0.81** | **$3,798** | **$5,263** |

Both tree-based models were tuned using 5-fold GridSearchCV.

### Cross-Validation

Mean CV R² ≈ 0.58.

Cross-validation is computed on log-transformed targets within the pipeline.  
Test metrics are evaluated in original USD scale for interpretability.

---

## Model Diagnostics

- Residual analysis shows no strong systematic bias.
- Predictions scale logically across low-risk and high-risk profiles.
- Feature importance confirms smoking status, BMI, and blood pressure as dominant drivers.

---

## Deployment

The final trained pipeline was serialized and deployed via Streamlit.

The application:

- Accepts real-time structured input  
- Applies preprocessing internally  
- Reverses log transformation correctly  
- Outputs predicted claim in USD  

🔗 **Live Demo:** https://health-insurance-claim-prediction-1-5bf4.onrender.com

---

## Project Structure

```
├── health_insurance_prediction.ipynb
├── insurance_claim_pipeline.pkl
├── app.py
├── insurance.csv
└── README.md
```

## Limitations
*   **Small Dataset:** Analysis is based on 1,332 usable records, which may limit the model's exposure to rare edge cases.
*   **Data Scope:** No historical claims data was included in this version.
*   **Target Variable:** The model is designed to predict the **expected claim amount** rather than the binary probability of a claim occurring.
*   **Generalization:** Cross-validation scores indicate moderate variance, suggesting sensitivity to specific data partitions.

## Future Improvements
- [ ] **Explainability:** Implement SHAP-based plots for individual prediction transparency.
- [ ] **Benchmarking:** Compare performance against LightGBM and CatBoost.
- [ ] **Uncertainty Mapping:** Integrate Quantile Regression to provide prediction intervals.
- [ ] **MLOps:** Implement drift monitoring for production environments.
- [ ] **Deployment:** Scale the current prototype into a robust REST API.

## Tech Stack

| Category | Tools |
| :--- | :--- |
| **Languages** | Python |
| **Data Handling** | pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit, Render |
| **Serialization** | joblib |

