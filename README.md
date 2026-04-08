# 📉 Telecom Customer Churn Prediction

**End-to-end machine learning project** to predict which telecom customers are likely to churn, enabling proactive retention strategies that protect revenue.

## 🎯 Business Problem

Customer churn is one of the most costly problems in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one. This project identifies high-risk customers **before** they leave, giving the business a window to intervene with targeted offers or outreach.

**Key business question:** *Which customers are most likely to cancel their subscription, and what factors drive that decision?*

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | IBM Sample Data — Telco Customer Churn (Kaggle) |
| Records | 7,043 customers |
| Features | 20 input features + 1 target (Churn) |
| Class Split | 73.5% No Churn / 26.5% Churn (imbalanced) |

**Features used:**
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Account info: Tenure, Contract Type, Payment Method, Paperless Billing
- Services: Phone, Internet, Streaming, Online Security, Tech Support
- Financials: Monthly Charges, Total Charges
- 

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn (Random Forest, SVM) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Model Serialization | Pickle |

---

## 🔁 Project Workflow

Raw Data (CSV)
      ↓
Data Cleaning & EDA
      ↓
Feature Engineering & Encoding
      ↓
Handle Class Imbalance (SMOTE)
      ↓
Model Training (Random Forest + SVM)
      ↓
Evaluation — AUC · F1 · Confusion Matrix · ROC Curve
      ↓
Model Export (Pickle)


## 🧹 Data Preprocessing

1. **Dropped** non-predictive column: `customerID`
2. **Detected and fixed** 11 blank string entries in `TotalCharges` — replaced with `0.0` and cast to float
3. **Standardized** categorical inconsistencies: `"No phone service"` and `"No internet service"` unified to `"No"` across 7 columns
4. **Binary encoding** of Yes/No columns — 12 columns converted to 0/1
5. **Label encoding** of Gender column
6. **One-Hot Encoding** for multi-class columns: `InternetService`, `Contract`, `PaymentMethod`
7. **MinMax Scaling** applied to numerical features: `tenure`, `MonthlyCharges`, `TotalCharges`

---

## 📈 Exploratory Data Analysis

Key patterns discovered:

- **Contract type** is the strongest churn predictor — month-to-month customers churn at ~3x the rate of two-year contract customers
- **Fiber optic internet** users churn significantly more than DSL users — likely driven by pricing sensitivity at higher monthly charges
- **Tenure** shows a clear inverse relationship with churn — customers who stay past 12 months are far less likely to leave
- **Monthly Charges** distribution confirms churners pay higher average bills
- **Electronic check** payment method carries the highest churn rate among all payment types

---

## ⚖️ Handling Class Imbalance

The dataset had a significant imbalance: 5,174 non-churners vs. 1,869 churners. Training on raw imbalanced data causes the model to be biased toward predicting "No Churn" always — inflating accuracy while missing actual churners.

**Solution: SMOTE (Synthetic Minority Over-sampling Technique)**
- Applied **only on the training set** — never on the test set (a common mistake that leaks data)
- After SMOTE: 4,138 non-churners vs. 4,138 churners in training data
- Ensures the model learns genuine churn patterns rather than majority-class bias


## 🤖 Model Results

| Model | Accuracy |

| Support Vector Machine (SVM) | 76.3% |
| **Random Forest (Best)** | **77.7%** |

**Random Forest selected** as the final model based on higher accuracy, built-in feature importance, and better interpretability for business stakeholders.


## 📊 Evaluation Metrics (Random Forest)

> **Accuracy alone is misleading on imbalanced datasets. AUC and Recall on the minority class are the primary evaluation metrics here.**

| Metric | No Churn | Churn |

| Precision | 0.86 | 0.57 |
| Recall | 0.83 | **0.64** |
| F1-Score | 0.85 | 0.60 |
| **AUC Score** | — | **0.8371** |

**What these numbers mean for business:**
- **AUC of 0.837** — the model correctly ranks a churner above a non-churner 84% of the time. Strong signal for prioritizing outreach lists.
- **Churn Recall of 64%** — the model catches 64% of actual churners. In retention strategy, missing a churner (false negative) costs more than a false alarm, so Recall on the churn class is the metric that matters most.
- **No Churn Precision of 0.86** — 86% of customers predicted to stay actually stay. Reliable for stable customer segmentation.

## 🔑 Top Churn Drivers (Feature Importance)

Based on Random Forest feature importance scores:

1. **Tenure** — longer-tenured customers are far less likely to churn
2. **Monthly Charges** — higher bills correlate strongly with churn risk
3. **Total Charges** — reflects billing history and tenure combined
4. **Contract: Month-to-month** — strongest contract-type churn signal
5. **Internet Service: Fiber optic** — premium service tier with highest churn rate


## 💡 Business Recommendations

Based on model findings, a retention team could:

1. **Target month-to-month customers** in their first 12 months with upgrade incentives to annual plans — highest churn risk window
2. **Flag fiber optic customers** paying above average monthly charges for proactive outreach
3. **Prioritize electronic check payers** for loyalty offers — highest churn rate by payment method
4. **Use the model's churn probability scores** to rank customers by risk and allocate retention budget efficiently — focus on the top 20% highest-risk customers first


## 📁 Project Structure

telecom-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── telecom_churn_analysis.ipynb    # Main notebook — EDA, modeling, evaluation
├── random_forest_model.pkl          # Saved trained model
└── README.md

## 🚀 How to Run

# Clone the repository
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Run the notebook
jupyter notebook telecom_churn_analysis.ipynb

## 📌 Future Improvements

- [ ] XGBoost / Gradient Boosting for performance comparison
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Streamlit dashboard for real-time churn probability prediction
- [ ] Threshold tuning — lower the classification threshold to improve Churn Recall further
- [ ] SHAP values for individual customer-level explainability

## 👤 Author

**Aquib Mansuri**
Data Analyst | Python · SQL · Power BI · Machine Learning
