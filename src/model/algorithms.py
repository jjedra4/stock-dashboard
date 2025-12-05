from .interface import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

class XGBoostModel(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model.load_model(path)

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

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_dim=10, hidden_dim=32, num_layers=1, epochs=10, lr=0.01, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = LSTMNet(input_dim, hidden_dim, 1, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _prepare_data(self, X, y=None):
        # Reshape for LSTM: (samples, time_steps, features)
        # For simplicity, assuming X is already 2D (samples, features) and we treat it as time_step=1
        # A real LSTM implementation needs a sequence generator (sliding window)
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1) 
        if y is not None:
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
            return X_tensor, y_tensor
        return X_tensor

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        self.model.train()
        X_tensor, y_tensor = self._prepare_data(X_train, y_train)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_data(X)
            preds = self.model(X_tensor)
            return preds.numpy().flatten()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
