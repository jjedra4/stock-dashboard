import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data.storage import SupabaseStorage

# Page config
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("🚀 Stock Prediction Dashboard")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Ticker", ["NVDA", "AAPL", "TSLA"])

# Load Data
@st.cache_data(ttl=3600)
def load_data(ticker):
    storage = SupabaseStorage()
    # This is a simplified query. You'll likely want to implement a specific method in SupabaseStorage
    # to fetch recent data for the dashboard.
    response = storage.client.table("stock_prices")\
        .select("*")\
        .eq("ticker", ticker)\
        .order("date", desc=True)\
        .limit(100)\
        .execute()
    
    if not response.data:
        return pd.DataFrame()
        
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data(ticker)

if not df.empty:
    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'))
    
    fig.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Recent Statistics")
    latest = df.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close", f"${latest['close']:.2f}")
    col2.metric("Volume", f"{latest['volume']:,}")
    # col3.metric("Predicted Next Day", "$...") # Placeholder

else:
    st.warning(f"No data found for {ticker}. Please run the ingestion pipeline.")

