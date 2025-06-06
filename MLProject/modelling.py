import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pathlib import Path
import os

def main():
    # Bersihkan run yg aktif
    if mlflow.active_run():
        mlflow.end_run()

    # 1. Load Data Hasil Preprocessing
    df = pd.read_csv('diabetes_processed.csv')

    # Pisahkan data train/test
    train_df = df[df['Data_Type'] == 'train']
    test_df = df[df['Data_Type'] == 'test']

    X_train = train_df.drop(['Outcome', 'Data_Type'], axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop(['Outcome', 'Data_Type'], axis=1)
    y_test = test_df['Outcome']

    # 2. Setup MLflow
    mlflow_dir = Path("mlruns").absolute()
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("GitHub_Actions_Diabetes")

    # 3. Train model dengan default parameters
    with mlflow.start_run(run_name="RF_Default_Params"):
        # Initialize and train model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # Log parameters and metrics
        mlflow.log_params(rf.get_params())
        mlflow.log_metrics(metrics)
        
        # Log model
        input_example = X_train.iloc[[0]]
        signature = infer_signature(input_example, rf.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="random_forest_model",
            signature=signature,
            input_example=input_example
        )
        
        # Print hasil
        print("\n=== Evaluation Metrics ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()