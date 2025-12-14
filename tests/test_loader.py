import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data.loader import update_stock_data

class TestLoader:
    
    @patch("src.data.loader.FMPClient")
    @patch("src.data.loader.SupabaseStorage")
    def test_update_stock_data(self, mock_storage_cls, mock_fmp_cls):
        """
        Test the update_stock_data logic which was previously in the Prefect flow.
        """
        # Mock dependencies
        mock_fmp = MagicMock()
        mock_fmp_cls.return_value = mock_fmp
        
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_latest_date.return_value = "2023-01-01"
        
        # Mock API response
        mock_df = pd.DataFrame({"close": [100, 101], "date": ["2023-01-02", "2023-01-03"]})
        mock_fmp.get_historical_price.return_value = mock_df
        
        # Run
        rows_added = update_stock_data("TEST")
        
        # Assertions
        assert rows_added == 2
        mock_storage.get_latest_date.assert_called_with("TEST")
        mock_fmp.get_historical_price.assert_called()
        mock_storage.upsert_stock_data.assert_called()
    
        # Verify ticker was added if missing (mock_df didn't have it)
        # The upsert call receives the df. We can check arguments.
        args, _ = mock_storage.upsert_stock_data.call_args
        df_passed = args[0]
        assert 'ticker' in df_passed.columns
        assert df_passed['ticker'].iloc[0] == "TEST"

    @patch("src.data.loader.FMPClient")
    @patch("src.data.loader.SupabaseStorage")
    def test_update_stock_data_no_new_data(self, mock_storage_cls, mock_fmp_cls):
        mock_fmp = MagicMock()
        mock_fmp_cls.return_value = mock_fmp
    
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_latest_date.return_value = "2023-01-01"
        
        # Empty response
        mock_fmp.get_historical_price.return_value = pd.DataFrame()
        
        rows_added = update_stock_data("TEST")
        
        assert rows_added == 0
        mock_storage.upsert_stock_data.assert_not_called()
