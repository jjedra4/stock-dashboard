import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.model.factory import ModelFactory
import os

def run_training_pipeline(
    data: pd.DataFrame, 
    target_col: str = 'close', 
    model_type: str = 'random_forest',
    model_params: dict = None,
    save_path: str = 'model.pkl'
):
    if model_params is None:
        model_params = {"n_estimators": 100, "random_state": 42}

    print(f"Starting training pipeline for {model_type}...")

    # 1. Feature Engineering (Simplified - assume data is ready)
    # Drop non-numeric columns if necessary or ensure they are handled
    features = data.drop(columns=[target_col, 'date', 'ticker'], errors='ignore')
    target = data[target_col]
    
    # 2. Train/Test Split (Time-based split is better for stocks!)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Instantiate Model via Factory
    model = ModelFactory.get_model(model_type, **model_params)
    
    # 4. Train
    model.train(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print(f"Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # 6. Save
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model, {"mae": mae, "mse": mse}

if __name__ == "__main__":
    # Example Test Run
    # Create dummy data
    import numpy as np
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'close': np.random.rand(100) * 100
    })
    
    run_training_pipeline(df, model_type="gradient_boosting", model_params={"n_estimators": 50})
