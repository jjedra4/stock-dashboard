import pandas as pd
import numpy as np
import os
from src.model.factory import ModelFactory
from src.data.loader import get_training_data
from dotenv import load_dotenv
from datetime import datetime

def train_final_model(ticker: str = "NVDA", save_path: str = "models"):
    """
    Trains the final model on the full dataset using the best hyperparameters found during tuning.
    """
    # 1. Load Data
    print(f"Loading data for {ticker}...")
    df = get_training_data(ticker)
    df = df.tail(2000)
    
    # Ensure sorted by date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # 2. Define Best Hyperparameters (from tuning)
    # jumping-glade-330
    params = {
        "colsample_bytree": 0.6074189350879698,
        "learning_rate": 0.0011385967158996672,
        "max_depth": 2,
        "n_estimators": 263,
        "subsample": 0.6938127165966703
    }
    
    print("Initializing model with best parameters...")
    model = ModelFactory.get_model("xgboost", **params)
    
    # 3. Train on FULL Data
    # The XGBoostModel.train method handles feature engineering internally.
    # It expects the raw DataFrame.
    print("Training on full dataset...")
    model.train(df)
    
    # 4. Save Model
    date = datetime.now().strftime("%Y%m%d")
    model_name = f"{ticker}_{date}_model.json"
    
    save_path = os.path.join(save_path, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    load_dotenv()
    train_final_model()
