import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import numpy as np
from src.data.storage import SupabaseStorage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Stock peer analysis dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Stock peer analysis

Easily compare stocks against others in their peer group.
"""

""  # Add some space.

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(tickers: list):
    storage = SupabaseStorage()
    all_prices = []
    all_preds = []
    
    for ticker in tickers:
        # Fetch latest prices first (desc=True) then sort
        # Limit 8000 ensures ~30 years of history (252 * 30 = 7560)
        response = storage.client.table("stock_prices")\
            .select("*")\
            .eq("ticker", ticker)\
            .order("date", desc=True)\
            .limit(8000)\
            .execute()
            
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            # Ensure tz-naive
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            # Sort ascending for charting
            df = df.sort_values('date')
            all_prices.append(df)
            
        # Fetch predictions
        response_pred = storage.client.table("predictions")\
            .select("*")\
            .eq("ticker", ticker)\
            .order("date", desc=True)\
            .limit(1000)\
            .execute()
    
        df_p = pd.DataFrame(response_pred.data)
        if not df_p.empty:
            df_p['date'] = pd.to_datetime(df_p['date'])
            if df_p['date'].dt.tz is not None:
                df_p['date'] = df_p['date'].dt.tz_localize(None)
            df_p = df_p.sort_values('date')
            all_preds.append(df_p)
            
    if all_prices:
        df_prices = pd.concat(all_prices)
    else:
        df_prices = pd.DataFrame()
        
    if all_preds:
        df_preds = pd.concat(all_preds)
    else:
        df_preds = pd.DataFrame()
        
    return df_prices, df_preds

def normalize_prices(df, start_date):
    # Normalize to 1.0 at start_date
    # For each ticker, find the price at start_date (or closest after)
    df_norm = df.copy()
    df_norm['normalized_price'] = np.nan
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask]
        # Find base price
        base_data = ticker_data[ticker_data['date'] >= start_date]
        if not base_data.empty:
            base_price = base_data.iloc[0]['close']
            df_norm.loc[mask, 'normalized_price'] = ticker_data['close'] / base_price
            
    return df_norm

# --- Layout ---
cols = st.columns([1, 3])

# --- Sidebar / Left Column ---
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

DEFAULT_STOCKS = ["NVDA", "AAPL", "MSFT"]
ALL_STOCKS = ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META"]

with top_left_cell:
    # Selectbox for stock tickers
    selected_tickers = st.multiselect(
        "Stock tickers",
        options=ALL_STOCKS,
        default=DEFAULT_STOCKS,
        placeholder="Choose stocks to compare. Example: NVDA",
    )

# Time horizon selector
horizon_map = {
    "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90),
    "6 Months": timedelta(days=180),
    "1 Year": timedelta(days=365),
    "5 Years": timedelta(days=365*5),
    "10 Years": timedelta(days=365*10),
    "All": None
}

with top_left_cell:
    # Pills for time range
    try:
        horizon = st.pills(
            "Time horizon",
            options=list(horizon_map.keys()),
            default="1 Year",
        )
    except AttributeError:
        horizon = st.radio(
            "Time horizon",
            options=list(horizon_map.keys()),
            index=3,
            horizontal=True,
        )

if not selected_tickers:
    top_left_cell.info("Pick some stocks to compare", icon=":material/info:")
    st.stop()

# --- Main Content / Right Column ---
right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

# Load data
with st.spinner("Loading data..."):
    df_prices, df_preds = load_data(selected_tickers)

if df_prices.empty:
    st.error("No data found.")
    st.stop()

# Filter Date Range
max_date = df_prices['date'].max()
if horizon_map[horizon]:
    start_date = max_date - horizon_map[horizon]
else:
    start_date = df_prices['date'].min()

df_display = df_prices[df_prices['date'] >= start_date].copy()

# Normalize
df_display = normalize_prices(df_display, start_date)

# Calculate Best/Worst
perf_stats = []
for t in selected_tickers:
    t_data = df_display[df_display['ticker'] == t]
    if not t_data.empty:
        start_price = t_data.iloc[0]['close']
        end_price = t_data.iloc[-1]['close']
        pct_change = (end_price - start_price) / start_price
        perf_stats.append({"ticker": t, "change": pct_change})

bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_left_cell:
    if perf_stats:
        best_stock = max(perf_stats, key=lambda x: x['change'])
        worst_stock = min(perf_stats, key=lambda x: x['change'])
        
        metric_cols = st.columns(2)
        metric_cols[0].metric(
            "Best stock",
            best_stock['ticker'],
            delta=f"{round(best_stock['change']*100)}%",
        )
        metric_cols[1].metric(
            "Worst stock",
            worst_stock['ticker'],
            delta=f"{round(worst_stock['change']*100)}%",
        )

# Plot normalized prices using Altair
with right_cell:
    # Prepare data for Altair (melt is handled by our structure, strictly speaking df_display is already long format with 'ticker' column)
    # We just need to rename columns to match standard expectation if needed, but 'ticker' works as color.
    
    chart_data = df_display[['date', 'ticker', 'normalized_price']].rename(columns={
        'date': 'Date', 
        'ticker': 'Stock', 
        'normalized_price': 'Normalized price'
    })
    
    st.altair_chart(
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            alt.X("Date:T"),
            alt.Y("Normalized price:Q").scale(zero=False),
            alt.Color("Stock:N"),
            tooltip=["Date", "Stock", "Normalized price"]
        )
        .properties(height=400)
        .interactive(), # Enables zoom/pan
        width="stretch"
    )

""
""

# --- Predictions Section ---
st.subheader("Latest Predictions")

if not df_preds.empty:
    latest_preds = []
    for t in selected_tickers:
        t_pred = df_preds[df_preds['ticker'] == t]
        if not t_pred.empty:
            latest = t_pred.iloc[-1]
            latest_preds.append({
                "Ticker": t,
                "Date": latest['date'].strftime("%Y-%m-%d"),
                "Predicted Return": f"{latest['predicted_pct_change']*100:.2f}%",
                "Signal": "Bullish 🟢" if latest['predicted_log_return'] > 0 else "Bearish 🔴"
            })
    
    if latest_preds:
        st.dataframe(pd.DataFrame(latest_preds), hide_index=True)
else:
    st.info("No predictions found for selected stocks.")
