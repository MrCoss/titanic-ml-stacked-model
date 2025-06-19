# train.py — Titanic Stacking Classifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Combine for feature engineering
data = pd.concat([train, test], sort=False)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Cabin'].fillna("U", inplace=True)

# Title from Name
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
                                       'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# FamilySize and IsAlone
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Age and Fare Bins
data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)

# Cabin initial and Ticket prefix
data['CabinInitial'] = data['Cabin'].str[0]
data['TicketPrefix'] = data['Ticket'].str.extract('^([A-Za-z./]+)', expand=False).fillna('None')

# Name length and Fare per person
data['NameLength'] = data['Name'].apply(len)
data['Fare_Per_Person'] = data['Fare'] / data['FamilySize']

# Encode categorical features
label = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title', 'CabinInitial', 'TicketPrefix']:
    data[col] = label.fit_transform(data[col])

# Sex * Pclass interaction
data['Sex_Pclass'] = data['Sex'] * data['Pclass']

# Feature list
features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title',
            'FamilySize', 'IsAlone', 'CabinInitial', 'Sex_Pclass',
            'TicketPrefix', 'NameLength', 'Fare_Per_Person']

# Prepare train/test data
X = data.loc[data['Survived'].notnull(), features]
y = train['Survived']
X_test_final = data.loc[data['Survived'].isnull(), features]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
cat = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, verbose=False, random_state=42)

# Meta model
meta_model = LogisticRegression(max_iter=1000)

# Stacking Ensemble
stack = StackingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('cat', cat)
], final_estimator=meta_model, cv=5)

# Train the model
stack.fit(X_train, y_train)

# Evaluate
val_preds = stack.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
report = classification_report(y_val, val_preds)
cv_scores = cross_val_score(stack, X, y, cv=5)
cv_mean = np.mean(cv_scores)

# Save model
joblib.dump(stack, "perfect_titanic_model.pkl")

# Show results
print("✅ Validation Accuracy:", round(val_acc, 4))
print("✅ Cross-validation Accuracy:", round(cv_mean, 4))
print("📊 Classification Report:\n", report)
