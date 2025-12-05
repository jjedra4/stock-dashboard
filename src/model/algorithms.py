from .interface import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd

class SklearnModel(BaseModel):
    """
    A wrapper for Scikit-Learn models.
    """
    def __init__(self, model_type: str = "random_forest", **params):
        self.model_type = model_type
        self.params = params
        self.model = self._get_sklearn_model()

    def _get_sklearn_model(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(**self.params)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(**self.params)
        elif self.model_type == "linear":
            return LinearRegression(**self.params)
        else:
            raise ValueError(f"Unsupported sklearn model type: {self.model_type}")

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

# Future: Add LSTMModel(BaseModel) here using PyTorch/TensorFlow

