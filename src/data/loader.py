import os
import pandas as pd
from datetime import datetime
from src.data.storage import SupabaseStorage
from src.data.fmp import FMPClient
from dotenv import load_dotenv

def _fetch_paginated_data(storage: SupabaseStorage, ticker: str, from_date: str = None) -> pd.DataFrame:
    """
    Fetches data from Supabase in chunks of 1000 rows to bypass API limits.
    """
    all_data = []
    offset = 0
    limit = 1000
    
    while True:
        print(f"Fetching chunk {offset}-{offset+limit}...")
        query = storage.client.table("stock_prices")\
            .select("*")\
            .eq("ticker", ticker)\
            .order("date", desc=False)\
            .range(offset, offset + limit - 1)
            
        if from_date:
            query = query.gt("date", from_date)
            
        response = query.execute()
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        
        if len(response.data) < limit:
            break
            
        offset += limit
        
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_training_data(ticker: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    """
    Fetches data from Supabase, caches it locally as CSV.
    1. Check cache.
    2. If valid, return.
    3. If outdated, fetch ONLY new rows from Supabase and append.
    4. If missing, fetch ALL rows.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"{ticker}.csv")
    
    storage = SupabaseStorage()
    
    # 1. Check if cache exists
    if os.path.exists(file_path):
        print(f"Checking cache for {ticker}...")
        df_cache = pd.read_csv(file_path)
        
        if "date" in df_cache.columns and not df_cache.empty:
            df_cache["date"] = pd.to_datetime(df_cache["date"])
            cache_latest = df_cache["date"].max().date()
            
            # Get Supabase latest date (cheap query)
            db_latest_str = storage.get_latest_date(ticker)
            
            if not db_latest_str:
                # DB empty, return cache
                return df_cache
                
            db_latest = pd.to_datetime(db_latest_str).date()
            
            if cache_latest >= db_latest:
                print("Cache is up to date with Supabase.")
                return df_cache
            else:
                print(f"Cache outdated (Cache: {cache_latest}, DB: {db_latest}). Fetching updates...")
                # Fetch only new data
                new_data = _fetch_paginated_data(storage, ticker, from_date=str(cache_latest))
                
                if not new_data.empty:
                    # Concatenate and deduplicate just in case
                    df_final = pd.concat([df_cache, new_data]).drop_duplicates(subset=["date", "ticker"]).sort_values("date")
                    df_final.to_csv(file_path, index=False)
                    print(f"Updated cache with {len(new_data)} new rows.")
                    return df_final
                return df_cache
    
    # 2. If no cache or empty, fetch everything
    print(f"Cache missing. Fetching all history for {ticker}...")
    df = _fetch_paginated_data(storage, ticker)
    
    if not df.empty:
        df.to_csv(file_path, index=False)
        print(f"Saved {len(df)} rows to cache.")
    else:
        print(f"No data found for {ticker}")
        
    return df

def update_stock_data(ticker: str) -> int:
    """
    Updates the Supabase database with the latest stock data from FMP.
    Checks the latest date in DB and only fetches new data.
    
    Returns:
        int: Number of new rows added.
    """
    print(f"Updating database for {ticker}...")
    storage = SupabaseStorage()
    client = FMPClient()
    
    # Get latest date from DB
    latest_date_str = storage.get_latest_date(ticker)
    from_date = latest_date_str if latest_date_str else (datetime.now() - pd.Timedelta(days=365*10)).strftime("%Y-%m-%d")
    
    # Fetch new data from FMP
    df_new = client.get_historical_price(ticker, from_date=from_date)
    
    if df_new.empty:
        print("No new data found from FMP.")
        return 0
        
    # Ensure ticker column exists
    if 'ticker' not in df_new.columns:
        df_new['ticker'] = ticker
    
    # We might get the same day if we fetch from latest_date, so we should handle duplicates
    # Upsert handles it if primary key is (ticker, date)
    # But FMP might return 'from_date' inclusive.
    # Let's trust upsert to handle overlap/updates.
    
    storage.upsert_stock_data(df_new, table_name="stock_prices")
    print(f"Successfully upserted {len(df_new)} rows to Supabase.")
    return len(df_new)

if __name__ == "__main__":
    load_dotenv()
    # df = get_training_data("NVDA")
    # print(df.tail())
    update_stock_data("NVDA")
