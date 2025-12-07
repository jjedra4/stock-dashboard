import pytest
import pandas as pd
import numpy as np
import os
import pickle
from unittest.mock import MagicMock, patch
from src.model.predict import load_model, make_predictions
from src.model.algorithms.xgboost_model import XGBoostModel

@pytest.fixture
def dummy_data():
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": np.random.uniform(100, 200, 50),
        "open": np.random.uniform(100, 200, 50),
        "high": np.random.uniform(100, 200, 50),
        "low": np.random.uniform(100, 200, 50),
        "volume": np.random.uniform(10000, 50000, 50),
        "vwap": np.random.uniform(100, 200, 50)
    })
    return df

class TestPredict:
    
    def test_load_model_pickle(self, tmp_path):
        # Create a dummy pickle file
        model_path = tmp_path / "model.pkl"
        dummy_obj = {"a": 1}
        with open(model_path, "wb") as f:
            pickle.dump(dummy_obj, f)
            
        loaded = load_model(str(model_path))
        assert loaded == dummy_obj

    def test_load_model_json_xgboost(self, tmp_path):
        # Create a dummy json file (content doesn't matter if we mock ModelFactory/load)
        model_path = tmp_path / "model.json"
        with open(model_path, "w") as f:
            f.write("{}")
            
        with patch("src.model.predict.ModelFactory") as mock_factory:
            mock_model = MagicMock()
            mock_factory.get_model.return_value = mock_model
            
            loaded = load_model(str(model_path), model_type="xgboost")
            
            assert loaded == mock_model
            mock_factory.get_model.assert_called_with("xgboost")
            mock_model.load.assert_called_with(str(model_path))

    def test_load_model_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_model("non_existent_file.pkl")

    def test_load_model_unsupported_format(self, tmp_path):
        path = tmp_path / "model.txt"
        with open(path, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_model(str(path))

    def test_make_predictions(self, dummy_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.2])
        
        # Test generic model
        preds = make_predictions(mock_model, dummy_data)
        assert len(preds) == 2
        mock_model.predict.assert_called_with(dummy_data)

    def test_make_predictions_xgboost_wrapper(self, dummy_data):
        # Test specific logic for XGBoost wrapper (checking if _prepare_data is called/handled)
        # We need a real XGBoostModel instance or a mock that looks like one
        mock_model = MagicMock(spec=XGBoostModel)
        # Set the 'model' attribute to exist so isinstance check passes if needed
        mock_model.model = MagicMock() 
        mock_model.model.predict.return_value = np.array([0.01] * len(dummy_data))
        
        # We need to mock _prepare_data to return a dummy DMatrix
        mock_model._prepare_data.return_value = "dummy_dmatrix"
        
        # We patch isinstance in predict.py OR we rely on the fact that if we pass an object, 
        # python's isinstance checks the class.
        # But make_predictions imports XGBoostModel. So we must use an instance of THAT class.
        
        # Let's create a real instance but mock its internals
        real_wrapper = XGBoostModel()
        real_wrapper.model = MagicMock()
        real_wrapper.model.predict.return_value = np.array([0.01])
        
        # Mock _prepare_data on the instance
        real_wrapper._prepare_data = MagicMock(return_value="dummy_dmatrix")
        
        preds = make_predictions(real_wrapper, dummy_data)
        
        real_wrapper._prepare_data.assert_called_with(dummy_data, train=False)
        real_wrapper.model.predict.assert_called_with("dummy_dmatrix")

