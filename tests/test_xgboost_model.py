import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
from unittest.mock import Mock, patch
from src.model.algorithms.xgboost_model import XGBoostModel

@pytest.fixture
def sample_data():
    # Create enough data to survive rolling windows (needs > 20 rows)
    dates = pd.date_range(start="2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'adjClose': prices,
        'volume': 100000,
        'vwap': prices
    })
    # Add target
    
    df.set_index('date', inplace=True)
    y = df['close'].shift(-1)
    return df, y

def test_init():
    model = XGBoostModel(n_estimators=50, learning_rate=0.1)
    assert model.params == {"n_estimators": 50, "learning_rate": 0.1}
    assert model.model is None

def test_prepare_data(sample_data):
    df, y = sample_data
    model = XGBoostModel()
    
    # Test with target
    dtrain = model._prepare_data(df, y)
    assert isinstance(dtrain, xgb.DMatrix)
    # Check if features were generated (we expect some rows dropped due to lags)
    assert dtrain.num_row() > 0
    assert dtrain.num_row() < 100  # dropped NaNs
    assert "log_ret" in dtrain.feature_names
    assert "day_sin" in dtrain.feature_names

    # Test inference mode (no target)
    dtest = model._prepare_data(df)
    assert isinstance(dtest, xgb.DMatrix)
    assert dtest.num_row() > 0

def test_train(sample_data):
    df, y = sample_data
    model = XGBoostModel(n_estimators=2, max_depth=2)
    
    model.train(df, y)
    
    assert model.model is not None
    assert isinstance(model.model, xgb.Booster)

def test_predict(sample_data):
    df, y = sample_data
    model = XGBoostModel(n_estimators=2, max_depth=2)
    model.train(df, y)
    
    # Prediction should return numpy array
    preds = model.predict(df)
    assert isinstance(preds, np.ndarray)
    assert len(preds) > 0
    assert len(preds) < 100 # due to feature engineering drops

def test_save_load(sample_data, tmp_path):
    df, y = sample_data
    model = XGBoostModel(n_estimators=2)
    model.train(df, y)
    
    save_path = str(tmp_path / "xgb_model.json")
    model.save(save_path)
    
    # Load new model
    loaded_model = XGBoostModel()
    loaded_model.load(save_path)
    
    assert loaded_model.model is not None
    assert isinstance(loaded_model.model, xgb.Booster)
    
    # Verify predictions match
    preds_orig = model.predict(df)
    preds_loaded = loaded_model.predict(df)
    np.testing.assert_array_almost_equal(preds_orig, preds_loaded)

