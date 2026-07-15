# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn using the **IBM Telco Customer Churn Dataset**. The project compares multiple classification models, selects the best-performing model based on **F1-Score**, explains predictions using **SHAP (SHapley Additive Explanations)**, and provides an interactive **Streamlit web application** for real-time customer churn prediction.

---

## Live Demo

🌐 **Deployed Streamlit App**

https://customer-churn-prediction-project-internship.streamlit.app/

---

# Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Retaining existing customers is often more cost-effective than acquiring new ones.

This project aims to predict whether a telecom customer is likely to discontinue the service by analyzing customer demographics, account information, subscribed services, billing details, and usage patterns.

The final solution combines:

- Machine Learning
- Explainable AI (XAI)
- Interactive Web Deployment

to create an end-to-end customer churn prediction system.

---

# Objectives

- Analyze customer churn patterns.
- Preprocess and clean the IBM Telco Customer Churn dataset.
- Perform feature engineering and data transformation.
- Train and compare multiple machine learning models.
- Evaluate model performance using multiple metrics.
- Select the best-performing model based on F1-Score.
- Explain model predictions using SHAP.
- Deploy the final model as an interactive Streamlit web application.

---

# Dataset

**Dataset:** IBM Telco Customer Churn Dataset (Kaggle)

**Target Variable**

- Churn (Yes / No)

### Dataset Statistics

- Total Records: **7,043**
- Features: **21 customer-related attributes**
- Binary Classification Problem

The dataset contains information about:

- Customer demographics
- Subscription details
- Internet services
- Account information
- Billing information
- Payment methods
- Contract type
- Customer churn status

---

# Project Workflow

The complete machine learning pipeline consists of:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Encoding Categorical Variables
6. Feature Scaling
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Model Comparison
11. Best Model Selection
12. SHAP Explainability
13. Streamlit Deployment

---

# Machine Learning Models Implemented

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Each model was evaluated on the same preprocessed dataset to identify the most suitable classifier for churn prediction.

---

# Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

Since the dataset is **imbalanced**, **F1-Score** was chosen as the primary evaluation metric because it provides a better balance between precision and recall.

---

# Best Model

After comparing all models, **XGBoost** achieved the best overall performance.

### Why XGBoost?

- Highest F1-Score
- Strong ROC-AUC performance
- Better generalization
- Robust ensemble learning approach
- Handles complex feature interactions effectively

Therefore, **XGBoost** was selected as the final production model used in the deployed application.

---

# Model Explainability (SHAP)

Machine learning predictions are often difficult to interpret.

To improve transparency, **SHAP (SHapley Additive Explanations)** was integrated into the project.

The deployed application provides explainability through:

- SHAP Feature Importance Plot
- SHAP Summary (Beeswarm) Plot
- SHAP Waterfall Plot
- Individual Prediction Explanation

This helps users understand **why** the model predicted whether a customer is likely to churn.

---

# Streamlit Web Application

The project is deployed as an interactive Streamlit application.

Users can:

- Enter customer details
- Predict customer churn
- View churn probability
- View customer risk level
- Understand prediction confidence
- Visualize SHAP explanations
- Explore dataset insights

---

# Features

- Interactive web interface
- Real-time churn prediction
- Customer risk analysis
- Probability gauge visualization
- SHAP Explainable AI
- Model insights dashboard
- Dataset information page
- Responsive Streamlit UI

---

# Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Plotly

### Deployment

- Streamlit
- Streamlit Community Cloud

### Model Serialization

- Joblib

---

# 📁 Project Structure

```
Customer-Churn-Prediction
│
├── app.py
├── config.py
├── preprocessing.py
├── utils.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── style.css
│
├── data/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── svm_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│
└── README.md
```

---

# Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/SadhikaRana/Customer-Churn-Prediction.git
```

### Navigate to the project directory

```bash
cd Customer-Churn-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# Application Output

The deployed application provides:

- Churn Prediction
- Churn Probability
- Risk Level
- Decision Threshold
- SHAP Waterfall Explanation
- Feature Importance
- Model Insights Dashboard

---

# Future Improvements

- Hyperparameter optimization
- Cross-validation pipeline
- Automated retraining
- Cloud database integration
- User authentication
- Batch prediction support
- REST API deployment
- Docker containerization

---

# Authors

Developed as part of a **Machine Learning Internship Project**.

Team Members:

- Reeva
- Sadhika
- Mugdha

---

# 📄 License

This project is developed for educational and learning purposes.
