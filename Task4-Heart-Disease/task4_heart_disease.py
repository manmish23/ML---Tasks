print("Task 4: Heart Disease Prediction Started...")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# 1. LOAD DATASET
import numpy as np  # Add this import at top with other imports

df = pd.read_csv('heart.csv')
df.replace('?', np.nan, inplace=True)  # Convert ? to NaN
df = df.dropna()  # Drop rows with missing values
df = df.astype(float)  # Convert all to float
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary target
print("Dataset Shape after cleaning:", df.shape)

# 2. EXPLORATORY DATA ANALYSIS
print("\nTarget distribution:")
print(df['target'].value_counts())
# 1 = Disease, 0 = No Disease

plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution: 0=No Disease, 1=Disease')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# 3. PREPARE DATA
X = df.drop('target', axis=1)
y = df['target']

# 4. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. FEATURE SCALING - Important for SVM and Logistic
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MODEL 1: LOGISTIC REGRESSION
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# 7. MODEL 2: RANDOM FOREST
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train) # RF doesn't need scaling
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 8. MODEL 3: SVM
print("\n--- SVM ---")
svm = SVC(probability=True, kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]

# 9. EVALUATION FUNCTION
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\nMetrics for {name}:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['No Disease', 'Disease']))

evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)
evaluate_model("SVM", y_test, y_pred_svm, y_prob_svm)

# 10. CONFUSION MATRIX - Best Model
best_model_name = "Random Forest"
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()

# 11. FEATURE IMPORTANCE - From Random Forest
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance - Random Forest')
plt.show()

print("\nTop 5 Important Features:\n", feature_importance.head())
print("\nTask 4 Complete ✅")