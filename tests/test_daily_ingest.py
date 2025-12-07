import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from prefect.testing.utilities import prefect_test_harness
from flows.daily_ingest import daily_ingest_flow, fetch_and_save_data, retrain_model, predict_next_day

@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    # Use Prefect test harness to use a temporary DB
    with prefect_test_harness():
        yield

class TestDailyIngestFlow:
    
    @patch("flows.daily_ingest.FMPClient")
    @patch("flows.daily_ingest.SupabaseStorage")
    def test_fetch_and_save_data_fn(self, mock_storage_cls, mock_fmp_cls):
        """
        Unit test for the underlying function of fetch_and_save_data task.
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
        
        # Call .fn() to bypass Prefect runtime for unit testing logic
        fetch_and_save_data.fn(ticker="TEST")
        
        # Assertions
        mock_storage.get_latest_date.assert_called_with("TEST")
        mock_fmp.get_historical_price.assert_called()
        mock_storage.upsert_stock_data.assert_called()
        
    @patch("flows.daily_ingest.train_final_model")
    def test_retrain_model_fn(self, mock_train):
        """
        Unit test for retrain_model task.
        """
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        
        result = retrain_model.fn(ticker="TEST")
        
        assert result == mock_model
        mock_train.assert_called_with(ticker="TEST", save_path="models")

    @patch("flows.daily_ingest.load_model")
    @patch("flows.daily_ingest.get_training_data")
    @patch("flows.daily_ingest.make_predictions")
    @patch("flows.daily_ingest.SupabaseStorage")
    def test_predict_next_day_fn(self, mock_storage_cls, mock_predict, mock_get_data, mock_load_model):
        """
        Unit test for predict_next_day task.
        """
        # Mock file existence to bypass os.path.exists check in the function?
        # The function uses os.path.exists. We can patch it or ensure path logic works.
        # Since we mock load_model, we just need execution to reach it.
        
        with patch("os.path.exists", return_value=True):
            # Setup data
            mock_df = pd.DataFrame({"close": range(100), "date": pd.date_range("2023-01-01", periods=100)})
            mock_get_data.return_value = mock_df
            
            # Setup predictions
            mock_predict.return_value = [0.01] # Log return
            
            # Run
            predict_next_day.fn(ticker="TEST")
            
            # Assertions
            mock_load_model.assert_called()
            mock_predict.assert_called()
            mock_storage_cls.return_value.upsert_stock_data.assert_called()

    @patch("flows.daily_ingest.fetch_and_save_data")
    @patch("flows.daily_ingest.retrain_model")
    @patch("flows.daily_ingest.predict_next_day")
    def test_daily_ingest_flow_execution(self, mock_predict, mock_retrain, mock_fetch):
        """
        Integration-like test running the flow in the test harness.
        """
        # Run the flow (Prefect test harness handles the orchestration context)
        daily_ingest_flow(tickers=["TEST"])
        
        # Verify tasks were called (mocked objects passed as tasks need careful checking)
        # Since we mocked the tasks themselves in the module, calling the flow should call the mocks.
        mock_fetch.assert_called_with("TEST")
        mock_retrain.assert_called_with("TEST")
        mock_predict.assert_called_with("TEST")


