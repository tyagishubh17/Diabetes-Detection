# DIABETES DETECTION (Using Pima Indians Diabetes Database)

A machine learning pipeline that predicts the onset of diabetes in patients using the Pima Indians Diabetes Database. This project trains and compares four classifiers — Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest — and lets you test predictions with your own patient profile.

---

## 📌 Problem Statement

Diabetes is a chronic metabolic disorder affecting millions worldwide. Early detection is critical to preventing severe complications. This project builds a supervised classification system that predicts whether a patient is diabetic or not, based on eight clinical diagnostic measurements.

---

## ✨ Features

- Automated dataset download (from a public GitHub profile) 
- Exploratory Data Analysis (EDA) with statistical summaries and correlation heatmap
- Data cleaning: replaces biologically invalid zero values with column medians
- Feature scaling using `StandardScaler`
- Trains and compares four ML models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
- Evaluation using Accuracy, Precision, Recall, and F1-score
- Confusion matrix visualizations for each model
- Best model auto-selection based on F1-score
- Interactive terminal mode for manual patient prediction

---

## 🛠️ Tech Stack

| Tool / Library    | Purpose                          |
|-------------------|----------------------------------|
| Python 3.x        | Core programming language        |
| Pandas            | Data loading and manipulation    |
| NumPy             | Numerical operations             |
| Scikit-learn      | ML models, scaling, evaluation   |
| Matplotlib        | Plotting confusion matrices      |
| Seaborn           | Correlation heatmap              |

---

## 📁 Project Structure

```
diabetes-detection/
├── app.py                  # Main script
├── diabetes.csv            # Dataset (auto-downloaded if missing)
├── requirements.txt        # Python dependencies
├── heatmap.png             # Correlation heatmap (generated)
├── confusion_matrices.png  # Confusion matrices for all models (generated)
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/tyagishubh17/Diabetes-Detection.git
cd diabetes-detection
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python app.py
```

The script will:
1. Download the dataset (if not present)
2. Run EDA and save `heatmap.png`
3. Clean and preprocess the data
4. Train all four models
5. Evaluate and print a performance comparison table
6. Save `confusion_matrices.png`
7. Prompt you to enter a custom patient profile for prediction

**Example manual input:**
```
Pregnancies: 2
Glucose: 138
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 47

-> Prediction Result: Diabetic (1)
```

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | ~75.3%   | ~0.67     | ~0.62  | ~0.64    |
| K-Nearest Neighbors | ~71.4%   | ~0.60     | ~0.67  | ~0.63    |
| Decision Tree       | ~71.4%   | ~0.62     | ~0.62  | ~0.62    |
| Random Forest       | ~73.4%   | ~0.63     | ~0.64  | ~0.63    |

> Logistic Regression achieved the best F1-score and is selected as the final model.

---

## 🔮 Future Scope

- Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- Cross-validation for more robust evaluation
- Add XGBoost or SVM classifiers
- Build a web-based prediction interface (Flask / Streamlit)
- Incorporate SHAP values for model explainability

---

## 👤 Author

**SHUBH TYAGI** <br>
Mail: tyagishubh.workspace@gmail.com
