# Predictive Modeling Using Machine Learning
### Bank Marketing Campaign — Term Deposit Subscription Prediction

## Project Overview
This project applies a complete, end-to-end machine learning workflow to predict whether a
bank client will subscribe to a term deposit as a result of a direct marketing (telemarketing)
campaign. Three classification models are trained, evaluated, and compared to identify the
best-performing approach.

## Problem Statement
Banks run marketing campaigns to promote term deposit products, but contacting every client is
costly and time-consuming. This project builds a predictive model that estimates the
likelihood of a client subscribing to a term deposit based on their personal, financial, and
campaign-related attributes — helping the bank target the clients most likely to convert.

This is a **binary classification** problem, with the target variable `y` indicating whether
the client subscribed (`yes`) or did not subscribe (`no`).

## Dataset Description
The dataset is the well-known **Bank Marketing** dataset (UCI Machine Learning Repository),
containing data from a Portuguese banking institution's direct marketing campaigns.

- **Rows:** 41,188 (41,176 after removing duplicates)
- **Columns:** 21 (20 features + 1 target)
- **Target variable:** `y` — has the client subscribed to a term deposit? (`yes` / `no`)

**Feature categories:**
| Group | Example Columns |
|---|---|
| Client demographics | `age`, `job`, `marital`, `education` |
| Financial status | `default`, `housing`, `loan` |
| Campaign contact info | `contact`, `month`, `day_of_week`, `duration`, `campaign` |
| Previous campaign outcome | `pdays`, `previous`, `poutcome` |
| Macroeconomic indicators | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

## Technologies Used
- **Python 3**
- **pandas** & **numpy** — data manipulation
- **matplotlib** & **seaborn** — data visualization
- **scikit-learn** — machine learning models & evaluation metrics
- **Jupyter Notebook** — interactive development environment

## Machine Learning Algorithms
Since the target variable is categorical, three classification models were built:

1. **Logistic Regression** — a linear baseline model, trained on scaled features.
2. **Decision Tree Classifier** — a non-linear model capturing feature interactions.
3. **Random Forest Classifier** — an ensemble of decision trees for improved accuracy and
   robustness against overfitting.

## Project Workflow
1. **Load Dataset** — read the CSV file into a pandas DataFrame.
2. **Data Preprocessing**
   - Replace `"unknown"` placeholder values with `NaN` and impute using the column mode.
   - Remove duplicate rows.
   - Label-encode all categorical columns.
   - Separate features (`X`) and target (`y`).
3. **Train-Test Split** — 80% training / 20% testing, stratified by target class.
4. **Model Building** — train Logistic Regression, Decision Tree, and Random Forest models.
5. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve
   and AUC score.
6. **Model Comparison** — consolidate all metrics into a single summary table and bar chart.
7. **Best Model Selection** — identify and justify the top-performing model.

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.9133** | 0.6450 | 0.5129 | **0.5714** | **0.9471** |
| Decision Tree | 0.9093 | 0.6171 | 0.5140 | 0.5608 | 0.8887 |
| Logistic Regression | 0.9092 | 0.6531 | 0.4138 | 0.5066 | 0.9318 |

## Results
**Random Forest** was the best-performing model, achieving the highest Accuracy (91.33%),
F1-Score, and ROC-AUC (0.947). This is because:

- As an **ensemble** of many decision trees, it reduces overfitting and variance compared to
  a single Decision Tree.
- It captures **non-linear relationships and feature interactions** that Logistic Regression,
  a purely linear model, cannot represent.
- It achieved the strongest ability to correctly rank positive vs. negative cases (highest
  ROC-AUC).

Feature importance analysis showed that call **duration**, the **euribor3m** interest rate,
and client **age** were the most influential predictors of subscription.

## Future Improvements
- Address class imbalance (only ~11% of clients subscribed) using techniques such as SMOTE or
  class-weighting to improve recall on the minority class.
- Perform hyperparameter tuning (e.g., `GridSearchCV`) for all models.
- Experiment with additional models such as Gradient Boosting or XGBoost.
- Apply cross-validation for more robust performance estimates.
- Perform feature engineering (e.g., binning `age`, interaction terms) to improve predictive
  power.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Predictive-Modeling-Using-Machine-Learning.git
   cd Predictive-Modeling-Using-Machine-Learning
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Predictive_Modeling.ipynb
   ```
4. Run all cells from top to bottom.

## Folder Structure
```
Predictive-Modeling-Using-Machine-Learning/
│── data/
│   └── dataset.csv
│── notebooks/
│   └── Predictive_Modeling.ipynb
│── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│── requirements.txt
│── README.md
```
