from ..interface import BaseModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

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
    def __init__(self, input_dim=10, hidden_dim=32, num_layers=1, epochs=10, lr=0.01, sequence_length=5, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        # Sequence length determines how many past steps to look at
        self.sequence_length = sequence_length 
        self.model = LSTMNet(input_dim, hidden_dim, 1, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _prepare_data(self, X, y=None):
        """
        Prepares data for LSTM.
        Ideally, this should create sliding windows.
        For now, keeping it simple as a 1-step lookback if data is just 2D.
        Enhancement: Implement sliding window here if X is raw time series.
        """
        # Ensure input is float32
        X_values = X.values.astype(np.float32)
        
        # Simple reshape: (samples, 1, features)
        # A proper LSTM implementation would create (samples, sequence_length, features) here
        X_tensor = torch.tensor(X_values).unsqueeze(1) 
        
        if y is not None:
            y_values = y.values.astype(np.float32)
            y_tensor = torch.tensor(y_values).view(-1, 1)
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

