import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv

load_dotenv()

class FMPClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY is not set")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_historical_price(self, symbol: str, from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """
        Fetch historical daily price data.
        """
        # Default to last 10 years if no date provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
        
        # FMP endpoint for historical price
        url = f"{self.base_url}/historical-price-full/{symbol}"
        params = {
            "apikey": self.api_key,
            "from": from_date,
            "to": to_date
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "historical" not in data or not data["historical"]:
            return pd.DataFrame()
            
        df = pd.DataFrame(data["historical"])
        df = df.drop(columns=['label'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['ticker'] = symbol
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df

if __name__ == "__main__":
    load_dotenv()
    # key = os.getenv("FMP_API_KEY")
    client = FMPClient()
    df = client.get_historical_price("NVDA", from_date="2025-10-01")
    print(df.head(10))
