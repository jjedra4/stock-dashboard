import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import MagicMock, patch
from src.model.train import train_final_model

@pytest.fixture
def dummy_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": np.random.uniform(100, 200, 100),
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(100, 200, 100),
        "low": np.random.uniform(100, 200, 100),
        "volume": np.random.uniform(10000, 50000, 100),
        "vwap": np.random.uniform(100, 200, 100)
    })
    return df

class TestTrain:
    
    @patch("src.model.train.get_training_data")
    @patch("src.model.train.ModelFactory")
    def test_train_final_model(self, mock_factory, mock_get_data, dummy_data, tmp_path):
        # Setup mocks
        mock_get_data.return_value = dummy_data
        
        mock_model = MagicMock()
        mock_factory.get_model.return_value = mock_model
        
        # Setup save path
        save_dir = tmp_path / "models"
        
        # Run training
        model = train_final_model(ticker="TEST", save_path=str(save_dir))
        
        # Assertions
        mock_get_data.assert_called_with("TEST")
        mock_factory.get_model.assert_called_once()
        mock_model.train.assert_called_once()
        mock_model.save.assert_called_once()
        
        # Check if directory was created (logic in function handles creation)
        assert os.path.exists(save_dir)

    def test_train_integration_xgboost(self, dummy_data, tmp_path):
        """
        Integration test with actual XGBoost model (using dummy data)
        """
        with patch("src.model.train.get_training_data", return_value=dummy_data):
            save_dir = tmp_path / "models"
            
            # This will actually train a small tree
            model = train_final_model(ticker="TEST", save_path=str(save_dir))
            
            # Verify we got an object back
            assert model is not None
            # Check if save was called (file might be named with timestamp, so checking dir)
            assert os.path.exists(save_dir)
            files = os.listdir(save_dir)
            assert len(files) > 0
            assert files[0].endswith(".json")

