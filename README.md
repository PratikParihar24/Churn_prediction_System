# 📡 Telco Customer Churn Prediction System

## 🎯 Business Objective
To identify high-risk customers likely to cancel their service (churn) and provide actionable retention strategies. This is a **Binary Classification** problem.

## 🛠️ Technical Highlights
* **Handling Class Imbalance:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target class (26% churn vs 74% non-churn).
* **Evaluation Strategy:** Prioritized **Recall** and **ROC-AUC** over simple Accuracy to ensure maximum detection of at-risk customers.
* **Architecture:** Followed a modular SDLC approach with a dedicated preprocessing engine and a Streamlit dashboard.

## 📊 Model Performance
* **Primary Model:** Random Forest Classifier (Optimized)
* **ROC-AUC Score:** ~0.84 (Highly effective at ranking risk)
* **Recall (Churn):** ~0.80 (Catches 80% of potential churners)

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Install as editable package: `pip install -e .`
3. Launch dashboard: `streamlit run app/main_dashboard.py`