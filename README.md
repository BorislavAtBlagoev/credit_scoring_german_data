# Credit Scoring Model – German Credit Data

This project predicts whether a customer is a **Good** or **Bad** credit risk based on historical data.

## 1. Objective

1. Use a public dataset (German Credit Data).
2. Perform data cleaning and preprocessing.
3. Train two classification models:
   - Logistic Regression  
   - Random Forest  
4. Explain which factors influence the prediction the most.

## 2. Dataset

The dataset contains customer demographic and financial information:

- Age, Sex, Job
- Housing type
- Saving and checking account status
- Credit amount and duration
- Loan purpose
- **Credit Risk** (1 = good, 2 = bad)

The target variable is converted to:
- **1 = good credit**
- **0 = bad credit**

## 3. Methods

### Preprocessing
- Missing values imputed  
- OneHotEncoder for categorical variables  
- No scaling (kept minimal per project requirements)  
- Train/test split 80/20  

### Models
- Logistic Regression
- Random Forest

### Evaluation Metrics
- Accuracy  
- Precision / Recall / F1  
- ROC AUC  

### Interpretation
- Logistic Regression coefficients  
- Random Forest feature importances  

These methods identify which factors contribute most strongly to credit risk (e.g., credit amount, duration, account status, purpose).

## 4. How to Run

Install dependencies:
```bash
pip install -r requirements.txt
