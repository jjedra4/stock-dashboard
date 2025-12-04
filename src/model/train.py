import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_model(data: pd.DataFrame, target_col: str = 'close'):
    """
    Basic training pipeline.
    """
    # 1. Prepare features
    # Assuming 'date' is index or dropped
    X = data.drop(columns=[target_col, 'date'], errors='ignore')
    y = data[target_col]
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model MAE: {mae}")
    
    return model

def save_model(model, path: str = "model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv(...)
    # model = train_model(df)
    # save_model(model)
    pass

