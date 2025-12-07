import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from src.model.tuning import HyperparameterTuner

# Helper to create dummy stock data
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
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    return df

class TestHyperparameterTuner:
    
    @patch("src.model.tuning.wandb.init")
    @patch("src.model.tuning.wandb.log")
    @patch("src.model.tuning.wandb.finish")
    def test_tuning_initialization_and_run(self, mock_finish, mock_log, mock_init, dummy_data):
        # Mock optuna study to avoid long running optimization
        with patch("optuna.create_study") as mock_create_study:
            mock_study = MagicMock()
            mock_study.best_params = {"n_estimators": 10, "max_depth": 3}
            mock_study.best_value = 0.05
            mock_create_study.return_value = mock_study
            
            # Setup Tuner
            tuner = HyperparameterTuner(dummy_data, model_type="xgboost", n_trials=1)
            
            # Run
            best_params = tuner.run()
            
            # Assertions
            assert best_params == {"n_estimators": 10, "max_depth": 3}
            mock_create_study.assert_called_once()
            mock_study.optimize.assert_called_once()

    def test_objective_function_structure(self, dummy_data):
        """
        Test that objective function returns a float (RMSE) and handles data splitting.
        """
        tuner = HyperparameterTuner(dummy_data, model_type="xgboost", n_trials=1)
        
        # Mock trial object
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 10
        mock_trial.suggest_float.return_value = 0.1
        
        # We need to mock wandb inside objective
        with patch("src.model.tuning.wandb.init"), \
             patch("src.model.tuning.wandb.log"), \
             patch("src.model.tuning.wandb.finish"):
            
            # Since dummy data ends in 2023, and split date is 2025 in tuning.py,
            # we need to adjust the split date logic OR mock the data to be recent.
            # Easier to update dummy data to be recent.
            recent_dates = pd.date_range(start="2025-01-01", periods=200, freq="D")
            recent_data = dummy_data.copy()
            recent_data['date'] = pd.concat([pd.Series(recent_dates[:100]), pd.Series(recent_dates[100:])], ignore_index=True)[:100]
            # Ensure we have enough data around the split point (2025-06-01)
            # 2025-01-01 + 200 days goes past June.
            recent_dates_long = pd.date_range(start="2025-01-01", periods=200, freq="D")
            long_df = pd.DataFrame({
                "date": recent_dates_long,
                "close": np.random.uniform(100, 200, 200),
                "open": np.random.uniform(100, 200, 200),
                "high": np.random.uniform(100, 200, 200),
                "low": np.random.uniform(100, 200, 200),
                "volume": np.random.uniform(10000, 50000, 200),
                "vwap": np.random.uniform(100, 200, 200)
            })
            
            tuner.data = long_df
            
            # Run objective
            rmse = tuner.objective(mock_trial)
            
            # Assert valid RMSE
            assert isinstance(rmse, float)
            assert rmse >= 0

    def test_data_validation(self, dummy_data):
        """Test that tuner raises error if split date is out of range"""
        tuner = HyperparameterTuner(dummy_data, model_type="xgboost")
        # dummy_data is 2023, split is 2025 -> should raise ValueError in objective
        
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 10
        mock_trial.suggest_float.return_value = 0.1
        
        with patch("src.model.tuning.wandb.init"):
             with pytest.raises(ValueError, match="Split date .* is after the last date"):
                 tuner.objective(mock_trial)

