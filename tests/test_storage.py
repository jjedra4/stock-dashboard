import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.data.storage import SupabaseStorage

@pytest.fixture
def mock_env_supabase(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test_key")

@pytest.fixture
def mock_supabase_client():
    with patch("src.data.storage.create_client") as mock_create:
        mock_client_instance = Mock()
        mock_create.return_value = mock_client_instance
        yield mock_client_instance

def test_init_error_without_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_KEY must be set"):
        SupabaseStorage()

def test_init_success(mock_env_supabase, mock_supabase_client):
    storage = SupabaseStorage()
    assert storage.url == "https://test.supabase.co"
    assert storage.key == "test_key"

def test_upsert_stock_data(mock_env_supabase, mock_supabase_client):
    storage = SupabaseStorage()
    
    # Create sample dataframe
    df = pd.DataFrame({
        "ticker": ["NVDA"],
        "date": ["2025-01-01"],
        "close": [100.0]
    })
    
    # Setup mock chain: client.table().upsert().execute()
    mock_table = Mock()
    mock_upsert = Mock()
    mock_execute = Mock()
    
    mock_supabase_client.table.return_value = mock_table
    mock_table.upsert.return_value = mock_upsert
    mock_upsert.execute.return_value = mock_execute
    
    # Call method
    response = storage.upsert_stock_data(df)
    
    # Assertions
    mock_supabase_client.table.assert_called_with("stock_prices")
    # Check if data was converted to records correctly
    expected_records = [{"ticker": "NVDA", "date": "2025-01-01", "close": 100.0}]
    mock_table.upsert.assert_called_with(expected_records)
    assert response == mock_execute

def test_get_latest_date_found(mock_env_supabase, mock_supabase_client):
    storage = SupabaseStorage()
    
    # Setup mock chain for select query
    mock_query = Mock()
    mock_response = Mock()
    mock_response.data = [{"date": "2025-01-05"}]
    
    mock_supabase_client.table.return_value = mock_query
    # Chaining methods: .select().eq().order().limit().execute()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value = mock_response
    
    result = storage.get_latest_date("NVDA")
    
    assert result == "2025-01-05"
    mock_supabase_client.table.assert_called_with("stock_prices")
    mock_query.eq.assert_called_with("ticker", "NVDA")

def test_get_latest_date_not_found(mock_env_supabase, mock_supabase_client):
    storage = SupabaseStorage()
    
    # Mock empty response
    mock_query = Mock()
    mock_response = Mock()
    mock_response.data = []  # No data found
    
    mock_supabase_client.table.return_value = mock_query
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value = mock_response
    
    result = storage.get_latest_date("UNKNOWN")
    
    assert result is None

