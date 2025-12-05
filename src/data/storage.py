import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

class SupabaseStorage:
    def __init__(self, url: str = None, key: str = None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
            
        self.client: Client = create_client(self.url, self.key)

    def upsert_stock_data(self, df: pd.DataFrame, table_name: str = "stock_prices"):
        """
        Upserts stock data into Supabase.
        Assumes DataFrame has columns matching the table schema.
        """
        # Convert DataFrame to list of dicts for Supabase
        # Handle date conversion to string if necessary
        records = df.to_dict(orient='records')
        
        # Supabase upsert
        # Ensure your table has a primary key (e.g., composite of ticker + date)
        response = self.client.table(table_name).upsert(records).execute()
        return response

    def get_latest_date(self, ticker: str, table_name: str = "stock_prices") -> str:
        """
        Get the latest date available for a ticker to know where to start fetching.
        """
        response = self.client.table(table_name)\
            .select("date")\
            .eq("ticker", ticker)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
            
        if response.data:
            return response.data[0]['date']
        return None

if __name__ == "__main__":
    pass