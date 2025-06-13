import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
from pathlib import Path

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
    mlflow.set_experiment("Diabetes_Prediction")
    
    # Autologging    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
        if sklearn_version.startswith('1.'):
            raise ImportError("Versi sklearn tidak kompatibel")
            
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            silent=True
        )
    except Exception as e:
        print(f"Autolog disabled: {str(e)}")
        # Fallback ke manual logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        })

    # 3. Train model dengan default parameters
    with mlflow.start_run(run_name="RF_Default_Params"):
        # Initialize and train model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Autolog
        rf.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = rf.predict(X_test)
        
        # Print hasil (opsional)
        print("\n=== Model Training Complete ===")

if __name__ == "__main__":
    main()