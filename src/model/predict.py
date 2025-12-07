import pandas as pd
import numpy as np
import os
import xgboost as xgb
import pickle
from src.model.factory import ModelFactory
from src.data.loader import get_training_data
from src.model.algorithms.xgboost_model import XGBoostModel # Explicit import for debug

def load_model(path: str = "model.pkl", model_type: str = "xgboost"):
    """
    Load a model from either a Pickle file or a JSON file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
        
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
        
    elif path.endswith(".json"):
        if model_type == "xgboost":
            model = ModelFactory.get_model("xgboost")
            model.load(path)
            return model
        else:
             raise ValueError(f"Loading JSON models for type '{model_type}' is not yet supported.")
    
    else:
        raise ValueError(f"Unsupported file format: {path}")

def make_predictions(model, data: pd.DataFrame):
    """
    Make predictions using the loaded model.
    """
    if isinstance(model, XGBoostModel):
        # 1. Feature Engineering
        dmat = model._prepare_data(data, train=False)
        
        # 2. Predict on DMatrix
        # This returns the raw predictions
        predictions = model.model.predict(dmat)
        
        return predictions
    else:
        # Fallback for other model types
        return model.predict(data)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    model_path = "models/NVDA_20251207_model.json"
    
    try:
        model = load_model(model_path, model_type="xgboost")
        
        raw_data = get_training_data("NVDA")
        input_data = raw_data.tail(200).copy() # Use enough history
        
        if 'date' in input_data.columns:
             input_data['date'] = pd.to_datetime(input_data['date'])
        
        preds = make_predictions(model, input_data)
        
        print("\nPredictions (Log Returns):")
        print(preds)
        
    except Exception as e:
        print(f"An error occurred: {e}")
