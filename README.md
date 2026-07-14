# Customer-Churn-Prediction
This project predicts customer churn using the IBM Telco Customer Churn dataset. Multiple machine learning models were trained and evaluated to identify best performing model.*SHAP* (SHapley Additive Explanations) was used to improve model interpretability.

# Objectives
-Analyze customer churn patterns.
-Preprocess and clean the IBM Telco Customer Churn dataset.
-Train and compare multiple machine learning models.
-Select the best-performing model using evaluation metrics.
-Explain model predictions using SHAP.
-Deploy the final model using Streamlit.

# Dataset
Dataset: Kaggle- IBM Telco Customer Churn Dataset
Target Variable: Churn (Yes/No)
- Records: 7,043
- Features: 21 customer-related features

#  Workflow 
-Data Collection
-Data Cleaning
-Feature Engineering
-Encoding & Scaling
-Train-Test Split
-Model Training
-Model Evaluation
-Model Comparison
-Best Model Selection 
-SHAP Explainability
-Streamlit Deployment

# Models Implemented
-Logistic Regression
-Decision Tree
-Support Vector Machine (SVM)
-Random Forest
-XGBoost

# Model Evaluation Metrics
The models were evaluated using:
-Accuracy
-Precision
-Recall
-F1-Score
-ROC-AUC
Since the dataset is imbalanced, F1-Score was used as the primary metric for selecting the best model.

# Best Model
XGBoost was selected as the final model because it achieved the *highest F1-Score* while maintaining strong ROC-AUC performance, making it the most suitable model for customer churn prediction.


# Model Explainability
SHAP (SHapley Additive Explanations) was used to interpret the predictions of the selected XGBoost model.
Generated visualizations include:
-SHAP Feature Importance
-SHAP Summary (Beeswarm) Plot
-SHAP Waterfall Plot
