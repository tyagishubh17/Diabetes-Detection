import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import urllib.request

def download_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Dataset not found. Downloading to {file_path}...")
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print(f"Dataset {file_path} found locally.")

def load_data(file_path):
    print(f"\nLoading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def perform_eda(df):
    print("\n--- Exploratory Data Analysis ---")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nNull Values Check:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Plot correlation heatmap
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("heatmap.png")
    plt.close()
    print("Saved correlation heatmap as heatmap.png")

def clean_data(df):
    print("\n--- Cleaning Data ---")
    # Biological features where 0 is structurally invalid (missing data)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        # Replace 0 with NaN so it's not counted in median, then replace NaN with median
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    print("Replaced 0 values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with column medians.")
    return df

def preprocess_data(df):
    print("\n--- Preprocessing Data ---")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for better readability internally if desired
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("Data split into training (80%) and testing (20%) sets and normalized using StandardScaler.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    print("\n--- Training Models ---")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Trained {name}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    print("\n--- Evaluating Models ---")
    results = []
    
    # Setup subplots for 4 confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1
        })
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f"Confusion Matrix: {name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.close()
    print("Saved all confusion matrices to confusion_matrices.png")
    
    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    
    # Custom format to prevent pandas display duplication bugs
    header = f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}"
    print(header)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1-score']:<10.4f}")
    
    return results_df

def select_best_model(results_df, models):
    # We choose F1-score as the primary metric for comparison here
    best_row = results_df.loc[results_df['F1-score'].idxmax()]
    best_model_name = best_row['Model']
    print("\n--- Best Model Selection ---")
    print(f"The best model is {best_model_name} with Accuracy {best_row['Accuracy']:.4f} and F1-score {best_row['F1-score']:.4f}.")
    
    return models[best_model_name], best_model_name

def predict_diabetes(model, scaler, input_data):
    # Create DataFrame from input to keep column names (warning-free scaling)
    input_df = pd.DataFrame([input_data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    # Scale user input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    print("\n--- Predicting New Samples ---")
    print("Input Profile:")
    print(input_df.to_string(index=False))
    
    if prediction == 1:
        print("-> Prediction Result: Diabetic (1)")
    else:
        print("-> Prediction Result: Not Diabetic (0)")

def main():
    file_path = "diabetes.csv"
    
    # 0. Download Dataset
    download_dataset(file_path)
    
    # 1. Load Data
    df = load_data(file_path)
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Clean Data
    df_cleaned = clean_data(df)
    
    # 4. Preprocess Data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_cleaned)
    
    # 5. Train Models
    models = train_models(X_train, y_train)
    
    # 6. Evaluate Models
    results_df = evaluate_models(models, X_test, y_test)
    
    # 7. Best Model Selection
    best_model, best_name = select_best_model(results_df, models)
    
    # 8. Interactive Testing Function
    print("\n" + "="*40)
    print("MANUAL TESTING MODE")
    print("="*40)
    while True:
        choice = input("Would you like to test the model with your own manually entered profile? (y/n): ").strip().lower()
        if choice != 'y':
            break
            
        try:
            print("\nPlease enter the patient's health metrics:")
            preg = float(input("Pregnancies (e.g., 1) [Range: 0 - 17]: "))
            gluc = float(input("Glucose (e.g., 85) [Range: 0 - 200]: "))
            bp = float(input("Blood Pressure (e.g., 66) [Range: 0 - 122]: "))
            skin = float(input("Skin Thickness (e.g., 29) [Range: 0 - 99]: "))
            ins = float(input("Insulin (e.g., 0) [Range: 0 - 846]: "))
            bmi = float(input("BMI (e.g., 26.6) [Range: 0.0 - 67.1]: "))
            dpf = float(input("Diabetes Pedigree Function (e.g., 0.351) [Range: 0.08 - 2.42]: "))
            age = float(input("Age (e.g., 31) [Range: 21 - 81]: "))
            
            user_sample = [preg, gluc, bp, skin, ins, bmi, dpf, age]
            predict_diabetes(best_model, scaler, user_sample)
            
        except ValueError:
            print("\nError: Invalid input. Please enter numerical values only.")
            
        print("-" * 40)
    
    print("\nPipeline execution finished successfully.")

if __name__ == "__main__":
    main()
