from src.data.fmp import FMPClient
from src.data.storage import SupabaseStorage
import pandas as pd

tickers = ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NFLX"]

if __name__ == "__main__":
    client = FMPClient()
    df = pd.DataFrame()
    for ticker in tickers:
        print(f"Loading {ticker} data...")
        df = pd.concat([df, client.get_historical_price(ticker, from_date="2000-01-01")])
    storage = SupabaseStorage()
    storage.upsert_stock_data(df, table_name="stock_prices")