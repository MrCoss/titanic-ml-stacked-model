# Titanic - Advanced Survival Prediction

An advanced solution to the **[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)** Kaggle competition. This project implements a sophisticated machine learning pipeline featuring detailed feature engineering and a **stacked ensemble** of powerful gradient boosting models to predict passenger survival.

-----

## Table of Contents

  - [1. Project Context & Objective](https://www.google.com/search?q=%231-project-context--objective)
  - [2. The Machine Learning Pipeline](https://www.google.com/search?q=%232-the-machine-learning-pipeline)
  - [3. Modeling Strategy: Stacked Ensemble](https://www.google.com/search?q=%233-modeling-strategy-stacked-ensemble)
  - [4. Performance Metrics](https://www.google.com/search?q=%234-performance-metrics)
  - [5. Project Structure Explained](https://www.google.com/search?q=%235-project-structure-explained)
  - [6. Technical Stack](https://www.google.com/search?q=%236-technical-stack)
  - [7. Local Setup & Execution Guide](https://www.google.com/search?q=%237-local-setup--execution-guide)
  - [8. Author & Contributions](https://www.google.com/search?q=%238-author--contributions)

-----

## 1\. Project Context & Objective

The sinking of the Titanic is one of the most infamous shipwrecks in history. The goal of this classic Kaggle competition is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data.

This project moves beyond basic models to demonstrate a robust, competitive data science workflow. The primary objective is to maximize predictive accuracy by using:

  - **Creative Feature Engineering:** To extract maximum signal from the available data.
  - **Advanced Modeling:** To employ a stacked ensemble of high-performing tree-based models.
  - **Rigorous Validation:** To ensure the model's performance is stable and generalizable.

-----

## 2\. The Machine Learning Pipeline

The project follows a structured workflow from data exploration to final submission.

### Step 1: Exploratory Data Analysis (EDA)

  - Analyzed the relationships between various features (e.g., `Pclass`, `Sex`, `Age`) and the `Survived` target variable.
  - Used visualizations to uncover key insights, such as the higher survival rates for women and passengers in first class.

### Step 2: Data Cleaning & Feature Engineering

  - **Data Cleaning:** Handled missing values in the `Age`, `Fare`, and `Embarked` columns using median imputation.
  - **Feature Engineering:** Created a rich set of new features to enhance model performance:
      - **Categorical Features:** Extracted `Title` (e.g., "Mr", "Miss", "Mrs") from `Name`, and `CabinInitial` from `Cabin`.
      - **Family Features:** Created `FamilySize` (from `SibSp` + `Parch`) and a binary `IsAlone` feature.
      - **Numerical Features:** Engineered `FarePerPerson` and created binned features for `Age` and `Fare`.
      - **Interaction Features:** Created combined features like `Sex_Pclass` to capture more complex patterns.

-----

## 3\. Modeling Strategy: Stacked Ensemble

To achieve high accuracy, this project uses a **stacked generalization ensemble**. Stacking combines multiple models to produce a "super model" that is often more performant than any single model alone.

### Level 0: Base Models

Four powerful and diverse tree-based models are trained on the data. Each model learns different patterns from the features.

1.  **Random Forest Classifier**
2.  **XGBoost Classifier**
3.  **LightGBM Classifier**
4.  **CatBoost Classifier**

### Level 1: Meta-Model

  - The predictions from the four base models are used as new features.
  - A simpler, final model—**Logistic Regression**—is trained on these "prediction features."
  - This meta-model learns the optimal way to weigh the predictions from the base models to make a final, more accurate prediction.

This two-level approach leverages the strengths of each algorithm, leading to a more robust and accurate final result.

-----

## 4\. Performance Metrics

The model's performance was evaluated using multiple metrics to ensure its reliability.

| Metric                     | Score       | Description                                                                  |
| -------------------------- | ----------- | ---------------------------------------------------------------------------- |
| **Validation Accuracy** | **84.36%** | Accuracy on a held-out local test set.                                       |
| **Cross-validation Score**| **83.39%** | Average accuracy from 5-fold cross-validation, indicating robust performance. |
| **Kaggle Public Score** | **0.76315** | The final score on the unseen Kaggle test dataset.                           |

-----

## 5\. Project Structure Explained

The repository is organized into modular scripts for a clean and reproducible workflow.

```
.
├── train.py                # Script to run the full training pipeline and save the stacked model.
├── test.py                 # Loads the saved model and generates predictions on test.csv.
├── compare.py              # Script for offline model validation and performance comparison.
├── submission.csv          # The final submission file generated by test.py.
├── titanic_model.pkl       # The serialized, trained stacked ensemble model.
├── data/                   # (Assumed) Contains train.csv, test.csv, etc.
├── requirements.txt        # A list of all Python dependencies.
└── README.md               # This detailed project documentation.
```

-----

## 6\. Technical Stack

  - **Core Language:** Python
  - **Data Handling:** Pandas, NumPy
  - **Machine Learning:** Scikit-learn
  - **Gradient Boosting Models:** XGBoost, LightGBM, CatBoost
  - **Model Persistence:** Joblib

-----

## 7\. Local Setup & Execution Guide

To replicate this project and generate the submission file, follow these steps.

### Step 1: Clone the Repository

```bash
git clone https://github.com/MrCoss/titanic-ml-stacked-model.git
cd titanic-ml-stacked-model
```

### Step 2: Create and Activate a Virtual Environment (Recommended)

```bash
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Stacked Model

This command runs the `train.py` script, which executes the entire training pipeline and saves the final model as `titanic_model.pkl`.

```bash
python train.py
```

### Step 5: Generate Test Predictions

This command runs `test.py`, which loads the saved model and creates the `submission.csv` file for Kaggle.

```bash
python test.py
```

-----

## 8\. Author & Contributions

  - **Author:** Costas Pinto
  - **Contributions:** Contributions are welcome\! Feel free to fork the project, improve the models or feature engineering, and submit a pull request.
