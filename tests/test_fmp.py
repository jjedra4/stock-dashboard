import pytest
from unittest.mock import patch, Mock
import pandas as pd
from src.data.fmp import FMPClient

@pytest.fixture
def mock_env_api_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test_key")

def test_init_without_api_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FMP_API_KEY is not set"):
        FMPClient()

def test_init_with_arg_key():
    client = FMPClient(api_key="provided_key")
    assert client.api_key == "provided_key"

def test_init_with_env_key(mock_env_api_key):
    client = FMPClient()
    assert client.api_key == "test_key"

@patch("src.data.fmp.requests.get")
def test_get_historical_price_success(mock_get, mock_env_api_key):
    # Setup mock response
    mock_response = Mock()
    mock_data = {
        "symbol": "NVDA",
        "historical": [
            {
                "date": "2023-10-27",
                "open": 405.0,
                "high": 410.0,
                "low": 400.0,
                "close": 408.0,
                "adjClose": 408.0,
                "volume": 1000000,
                "unadjustedVolume": 1000000,
                "change": 3.0,
                "changePercent": 0.7,
                "vwap": 405.5,
                "label": "October 27, 23",
                "changeOverTime": 0.007
            },
            {
                "date": "2023-10-26",
                "open": 400.0,
                "high": 405.0,
                "low": 395.0,
                "close": 401.0,
                "adjClose": 401.0,
                "volume": 900000,
                "unadjustedVolume": 900000,
                "change": 1.0,
                "changePercent": 0.25,
                "vwap": 400.5,
                "label": "October 26, 23",
                "changeOverTime": 0.0025
            }
        ]
    }
    mock_response.json.return_value = mock_data
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = FMPClient()
    df = client.get_historical_price("NVDA")

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # Check if sorted by date (FMP returns desc, our code sorts asc)
    assert df.iloc[0]["date"] < df.iloc[1]["date"]
    
    # Verify API call parameters
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert "NVDA" in args[0]
    assert kwargs["params"]["apikey"] == "test_key"

@patch("src.data.fmp.requests.get")
def test_get_historical_price_empty(mock_get, mock_env_api_key):
    mock_response = Mock()
    mock_response.json.return_value = {"symbol": "NVDA", "historical": []}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = FMPClient()
    df = client.get_historical_price("NVDA")

    assert isinstance(df, pd.DataFrame)
    assert df.empty

