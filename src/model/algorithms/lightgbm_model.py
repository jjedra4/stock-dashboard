from ..interface import BaseModel
import lightgbm as lgb
import pandas as pd
import pickle

class LightGBMModel(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = lgb.LGBMRegressor(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
            