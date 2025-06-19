import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load data
gender = pd.read_csv("data/gender_submission.csv")
submission = pd.read_csv("submission.csv")

# Merge to compare
compare = pd.merge(submission, gender, on='PassengerId', suffixes=('_pred', '_true'))

# Evaluate
y_true = compare['Survived_true']
y_pred = compare['Survived_pred']

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"✅ Offline Accuracy: {accuracy:.4f}\n")
print("📊 Full Classification Report:")
print(report)
