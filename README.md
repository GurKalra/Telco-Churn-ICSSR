# Telco Customer Churn Prediction (TRACK-1)

## Project Overview
**Objective:** Predict whether a telecommunications customer will leave (churn) based on their demographic and service usage data.

**Dataset Provided:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

This project implements a Machine Learning pipeline to identify at-risk customers. It compares a **Baseline Model (Logistic Regression)** against an **Improved Model (XGBoost)**, performing a detailed error analysis to select the best performer.

---
## Methodology & Data Splitting

### Data Preparation
* **Encoding:** Converted `Churn` to binary (1/0) and used One-Hot Encoding for categorical features.
* **Split Strategy:** Used an **80/20 split** (80% Training, 20% Testing) with a fixed random seed for reproducibility.

### Split Code
```python
from sklearn.model_selection import train_test_split

#Converting churn to yes:1 and no:0
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

#Converting all other data to numeric values
df_encoded = pd.get_dummies(df, drop_first=True)

# Setting features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

#Spliiting data (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print("Data successfully split!")
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")
```

### Output
```Plaintext
Data successfully split!
Training Data Shape: (5625, 30)
Testing Data Shape: (1407, 30)
```
---
## Model Training

### 1. Baseline Model: Logistic Regression
I started with simple linear model to establish a performance baseline.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize and training the model
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

# making predictions  
y_pred = model_lr.predict(X_test)

# performancce evaluation
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
### Baseline Results: 
<img width="467" height="295" alt="Screenshot_20260211_003703" src="https://github.com/user-attachments/assets/905b2778-1bcb-4df4-a550-7fc35b2192a9" />

### 2. Improved Model: XGBoost (Tuned)
I used XGBoost with hyperparameter tuning to capture non-linear patterns and reduce overfitting
```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# initializing and training xgboost
model_xbg_tuned = XGBClassifier(
    n_estimators = 100, 
    learning_rate = 0.05, 
    max_depth = 3,
    subsample = 0.8,
    eval_metric = 'logloss',
    use_label_encoder = False,
    random_state = 42
)
model_xbg_tuned.fit(X_train, y_train)

#prediction
y_pred_tuned = model_xbg_tuned.predict(X_test)

#results
print("--- XGBoost Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_tuned):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))
```

### XGBoost Results: 
<img width="460" height="320" alt="Screenshot_20260211_003906" src="https://github.com/user-attachments/assets/4c713ff9-1f40-450e-aafe-19f61ad86dd8" />

---
## Error Analysis & Evaluation
To scientifically compare the models, I evaluated them on **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

## Evaluation Code
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# getting all metrics
def get_metrics(y_true, y_pred, y_prob, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob)
    }

# Get probabilities for ROC curve
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
y_prob_xgb = model_xbg_tuned.predict_proba(X_test)[:, 1]

# Creating the Comparison Table
metrics_lr = get_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
metrics_xgb = get_metrics(y_test, y_pred_xgb, y_prob_xgb, "XGBoost (Tuned)")
df_compare = pd.DataFrame([metrics_lr, metrics_xgb])

# Plotting 
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot A: Confusion Matrix (Baseline)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Baseline (LogReg) Mistakes')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# Plot B: Confusion Matrix (Improved)
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Improved (XGBoost) Mistakes')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

# Plot C: ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

axes[2].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {metrics_lr["AUC-ROC"]:.2f})', linestyle='--')
axes[2].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {metrics_xgb["AUC-ROC"]:.2f})', linewidth=2)
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[2].set_title('ROC Curve Comparison')
axes[2].set_xlabel('False Positive Rate'); axes[2].set_ylabel('True Positive Rate')
axes[2].legend()

plt.tight_layout()
plt.show()

# Final Table
print("\n--- Final Model Comparison Table ---")
print(df_compare.round(4).to_string(index=False))
```
### Visual Analysis:
<img width="1598" height="534" alt="Screenshot_20260211_004122" src="https://github.com/user-attachments/assets/f90b8b15-81c4-4a22-8019-c54058f82d06" />

### Interpretation of the Charts:
  #### 1. Confusion Matrix (Left vs. Center)
  * **False Positives (Top Right Box)**: XGBoost reduced these from 118 to 99. This means the model is         more "trustworthy" (Higher Precision) and won't annoy happy customers with unnecessary retention           offers.
  * **False Negatives (Bottom Left Box)**: Similar performance, with XGBoost being slightly more conservative.
  #### 2. ROC Curve (Right):
  * The **Orange Line (XGBoost)** is consistently above the **Blue Line (Baseline)**.
  * **AUC Score**: XGBoost achieved 0.84, which is considered excellent discrimination power.

---
## Final Verdict: The Best Model
**Winner: XGBoost (Tuned)**
* While the accuracy gain is modest, the significant improvement in Precision (+3%) is critical for business impact. A higher precision means that when the AI predicts a customer will churn, it is much more likely to be correct, ensuring that retention budgets are spent efficiently.
---

## Repository Contents
* ```churn_analysis.ipynb```: Complete notebook.
* ```churn_model.pkl```: Saved XGBoost model file.
* ```model_features.pkl```: Feature column names.
