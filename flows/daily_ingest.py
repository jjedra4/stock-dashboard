from prefect import flow, task
import pandas as pd
from datetime import datetime
from src.data.fmp import FMPClient
from src.data.storage import SupabaseStorage
from src.features.indicators import add_technical_indicators

@task(retries=3)
def fetch_data(ticker: str):
    client = FMPClient()
    # Fetch last 100 days to ensure we have enough for indicators
    df = client.get_historical_price(ticker, from_date=(datetime.now() - pd.Timedelta(days=200)).strftime("%Y-%m-%d"))
    return df

@task
def process_data(df: pd.DataFrame):
    if df.empty:
        return df
    df = add_technical_indicators(df)
    # Drop rows with NaN indicators if you want clean data for storage
    df = df.dropna()
    return df

@task
def save_to_db(df: pd.DataFrame, ticker: str):
    if df.empty:
        print("No data to save.")
        return
    
    storage = SupabaseStorage()
    # Add ticker column if missing
    if 'ticker' not in df.columns:
        df['ticker'] = ticker
        
    # Convert date to string for JSON serialization if needed, or keep as is if Supabase client handles it
    # df['date'] = df['date'].astype(str)
    
    storage.upsert_stock_data(df, table_name="stock_prices")
    print(f"Saved {len(df)} rows for {ticker}")

@flow(name="Daily Stock Ingest")
def daily_ingest_flow(tickers: list = ["NVDA"]):
    for ticker in tickers:
        data = fetch_data(ticker)
        processed_data = process_data(data)
        save_to_db(processed_data, ticker)

if __name__ == "__main__":
    daily_ingest_flow()

