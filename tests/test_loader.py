import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, call
from src.data.loader import get_training_data, _fetch_paginated_data

# --- Fixtures ---
@pytest.fixture
def mock_storage_client():
    with patch("src.data.loader.SupabaseStorage") as MockStorage:
        mock_instance = MockStorage.return_value
        mock_instance.client.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = []
        yield mock_instance

@pytest.fixture
def temp_cache_dir(tmp_path):
    return str(tmp_path / "data/cache")

# --- Helper Test: _fetch_paginated_data ---

def test_fetch_paginated_data_one_page(mock_storage_client):
    # Setup mock to return 1 page of data (e.g. 500 rows)
    mock_data = [{"date": "2025-01-01", "ticker": "NVDA", "close": 100}] * 500
    
    # Mock chain
    mock_query = mock_storage_client.client.table().select().eq().order()
    mock_query.range.return_value.execute.return_value.data = mock_data
    # Second call returns empty to stop loop
    # We need to simulate: 
    # 1. First call returns 500 items (which is < limit 1000, so loop breaks naturally in code)
    # If loop breaks naturally because len < limit, it won't call again.
    # To force a second call, first batch must be >= limit (1000).
    
    full_page_data = [{"date": "2025-01-01", "ticker": "NVDA", "close": 100}] * 1000
    
    mock_query.range.return_value.execute.side_effect = [
        Mock(data=full_page_data), 
        Mock(data=[])
    ]

    df = _fetch_paginated_data(mock_storage_client, "NVDA")
    
    assert len(df) == 1000
    # Now it should be called twice (one full page, one check next page)
    assert mock_query.range.call_count == 2 

def test_fetch_paginated_data_empty(mock_storage_client):
    mock_storage_client.client.table().select().eq().order().range().execute.return_value.data = []
    df = _fetch_paginated_data(mock_storage_client, "NVDA")
    assert df.empty

# --- Main Test: get_training_data ---

def test_get_training_data_no_cache_fetches_all(mock_storage_client, temp_cache_dir):
    ticker = "NVDA"
    mock_data = [{"date": "2025-01-01", "ticker": ticker, "close": 100}]
    
    # Mock _fetch_paginated_data internal logic or just mock the storage response
    # Let's rely on the storage mock we set up
    mock_query = mock_storage_client.client.table().select().eq().order()
    # Return data then empty
    mock_query.range.return_value.execute.side_effect = [Mock(data=mock_data), Mock(data=[])]
    
    df = get_training_data(ticker, cache_dir=temp_cache_dir)
    
    assert len(df) == 1
    assert os.path.exists(os.path.join(temp_cache_dir, f"{ticker}.csv"))

def test_get_training_data_cache_valid(mock_storage_client, temp_cache_dir):
    ticker = "NVDA"
    cache_file = os.path.join(temp_cache_dir, f"{ticker}.csv")
    os.makedirs(temp_cache_dir, exist_ok=True)
    
    # Create dummy cache
    df_cache = pd.DataFrame({"date": ["2025-01-01"], "ticker": [ticker], "close": [100]})
    df_cache.to_csv(cache_file, index=False)
    
    # Mock DB saying latest date is same as cache
    mock_storage_client.get_latest_date.return_value = "2025-01-01"
    
    df = get_training_data(ticker, cache_dir=temp_cache_dir)
    
    # Should load from cache, NOT call fetch_paginated (no query.range calls)
    # Note: storage is instantiated, so we check if it tried to fetch new data
    mock_storage_client.client.table.assert_not_called() 
    assert len(df) == 1

def test_get_training_data_cache_outdated(mock_storage_client, temp_cache_dir):
    ticker = "NVDA"
    cache_file = os.path.join(temp_cache_dir, f"{ticker}.csv")
    os.makedirs(temp_cache_dir, exist_ok=True)
    
    # Cache has data up to Jan 1
    df_cache = pd.DataFrame({"date": ["2025-01-01"], "ticker": [ticker], "close": [100]})
    df_cache.to_csv(cache_file, index=False)
    
    # DB has data up to Jan 2
    mock_storage_client.get_latest_date.return_value = "2025-01-02"
    
    # Mock fetch response for the NEW data
    # Ensure consistent types. Date should be string from API, converted to datetime in loader
    new_data = [{"date": "2025-01-02", "ticker": ticker, "close": 105}]
    
    # The query includes .gt("date", ...) now
    mock_query = mock_storage_client.client.table().select().eq().order()
    
    # Setup the mock chain to handle .gt().range().execute()
    # Important: The code calls query.gt(...).range(...).execute()
    # So we need the return of gt() to be the object that has range()
    mock_gt_result = Mock()
    mock_query.gt.return_value = mock_gt_result
    
    mock_range_result = Mock()
    mock_gt_result.range.return_value = mock_range_result
    
    mock_range_result.execute.side_effect = [
        Mock(data=new_data), 
        Mock(data=[])
    ]
    
    # We need to patch _fetch_paginated_data call or ensure mock structure supports .gt()
    # Since .gt is called on the query object, we need to support it in mock
    # The code calls: query = query.gt("date", from_date) -> query.range(...)
    
    # Let's rely on the mocked storage client structure we built above instead of patching the function
    # This tests the integration of logic better
    
    # CAUTION: The loader converts cache 'date' to datetime. 
    # concat will mix datetime (from cache) and string (from new_data dict if not converted).
    # But loader.py: _fetch_paginated_data converts to datetime before returning!
    # So we need to ensure _fetch_paginated_data logic is respected or mocked to return DF with datetime.
    
    with patch("src.data.loader._fetch_paginated_data") as mock_fetch:
        # Mock must return DataFrame with datetime objects, not strings
        df_new = pd.DataFrame(new_data)
        df_new["date"] = pd.to_datetime(df_new["date"])
        mock_fetch.return_value = df_new
        
        df = get_training_data(ticker, cache_dir=temp_cache_dir)
        
        # Verify result is concatenated
        assert len(df) == 2
        assert df.iloc[1]["date"] == pd.Timestamp("2025-01-02")
        
        # Verify file was updated
        df_saved = pd.read_csv(cache_file)
        assert len(df_saved) == 2

