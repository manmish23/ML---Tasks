print("Script started...")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv('german_credit_data.csv')
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())

# 2. Your kredit column is ALREADY 1=Good, 0=Bad
print("\nUnique values in 'kredit' column:", df['kredit'].unique())
df['Risk'] = df['kredit']  # No mapping needed
df = df.drop('kredit', axis=1)

print("\nTarget distribution:")
print(df['Risk'].value_counts())

# 3. All columns are already numeric in UCI format
X = df.drop('Risk', axis=1)
y = df['Risk'].astype(int)

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model 1: Logistic Regression
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# 7. Model 2: Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 8. Evaluation Function
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\nMetrics for {name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['Bad', 'Good']))

evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# 9. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# 10. Feature Importance
feature_importance = pd.DataFrame({'Feature': df.drop('Risk', axis=1).columns, 'Importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nTop 5 Important Features:\n", feature_importance.head())