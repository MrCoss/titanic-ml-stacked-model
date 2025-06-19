import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load data
test = pd.read_csv("data/test.csv")
model = joblib.load("titanic_model.pkl")

# Combine with dummy survived column for feature consistency
test['Survived'] = None
data = test.copy()

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Cabin'].fillna("U", inplace=True)

# Feature engineering
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                       'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)
data['CabinInitial'] = data['Cabin'].str[0]
data['TicketPrefix'] = data['Ticket'].str.extract('^([A-Za-z./]+)', expand=False).fillna('None')
data['NameLength'] = data['Name'].apply(len)
label = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title', 'CabinInitial', 'TicketPrefix']:
    data[col] = label.fit_transform(data[col])
data['Sex_Pclass'] = data['Sex'] * data['Pclass']
data['Fare_Per_Person'] = data['Fare'] / data['FamilySize']

# Final feature list (must match training)
features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title',
            'FamilySize', 'IsAlone', 'CabinInitial', 'Sex_Pclass',
            'TicketPrefix', 'NameLength', 'Fare_Per_Person']

X_test = data[features]

# Predict
predictions = model.predict(X_test)

# Save submission file
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions.astype(int)
})
submission.to_csv("submission.csv", index=False)
print("✅ Predictions saved to submission.csv")
