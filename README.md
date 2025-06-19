# 🚢 Titanic - Machine Learning from Disaster

Predict survival on the Titanic using a powerful ensemble of ML models including Random Forest, XGBoost, LightGBM, and CatBoost with advanced feature engineering.

---

## 📌 Project Overview

This project is a solution to the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition hosted on Kaggle. It demonstrates a complete pipeline including:

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Data Cleaning & Feature Engineering
- 🤖 Model Building (Stacked Ensemble)
- 📈 Evaluation & Cross-validation
- 🧪 Test Predictions & Kaggle Submission

---

## 🧠 Models Used

- **Random Forest**
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **Logistic Regression (as meta-model in stacking)**

---

## 🧪 Features Used

- `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`
- Extracted `Title` from `Name`
- `FamilySize`, `IsAlone`, `FarePerPerson`
- `CabinInitial`, `TicketPrefix`, `AgeBin`, `FareBin`
- `Sex_Pclass`, `NameLength`

---

## 📊 Model Performance

| Metric                     | Score     |
|---------------------------|-----------|
| ✅ Validation Accuracy     | 84.36%    |
| ✅ Cross-validation Score  | 83.39%    |
| 🧪 Kaggle Public Score     | 0.76315   |

---

## 📁 File Structure

```

.
├── train.py         # Main training pipeline with stacking ensemble
├── test.py          # Generates test predictions (submission.csv)
├── compare.py       # Offline validation & performance metrics
├── submission.csv   # Submission file for Kaggle
├── titanic\_model.pkl # Saved model
├── train.csv        # Training data
├── test.csv         # Test data
├── gender\_submission.csv # Sample submission
└── README.md        # Project readme (this file)

````

---

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/MrCoss/titanic-ml-stacked-model.git
cd titanic-ml-stacked-model
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train.py
```

4. Generate predictions:

```bash
python test.py
```

---

## 🤝 Contributions

Feel free to fork the project and improve it. PRs are welcome!


> Created with ❤️ by [Costas Pinto](https://www.kaggle.com/mrcoss)

```
