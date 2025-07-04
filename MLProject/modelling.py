import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
from pathlib import Path

def main():
    # Bersihkan run yg aktif
    if mlflow.active_run():
        mlflow.end_run()

    # Load Data Hasil Preprocessing
    df = pd.read_csv('diabetes_processed.csv')

    # Pisahkan data train/test
    train_df = df[df['Data_Type'] == 'train']
    test_df = df[df['Data_Type'] == 'test']

    X_train = train_df.drop(['Outcome', 'Data_Type'], axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop(['Outcome', 'Data_Type'], axis=1)
    y_test = test_df['Outcome']

    # Setup MLflow
    mlflow_dir = Path("mlruns").absolute()
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("Diabetes_Prediction")

    # Autolog
    mlflow.sklearn.autolog()

    # Train model
    try:
        with mlflow.start_run(run_name="RF_Default_Params"):
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            print("\n=== Model Training Complete ===")
    except Exception as e:
        print(f"Error starting MLflow run: {e}")

if __name__ == "__main__":
    main()