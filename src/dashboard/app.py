import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from src.data.storage import SupabaseStorage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Stock Peer Analysis", layout="wide")

# Force Dark Theme CSS (Optional, Streamlit detects system theme usually)
# But we can hint users to switch to dark theme in settings or inject CSS.
# For now, we rely on Streamlit's theming.

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(tickers: list):
    storage = SupabaseStorage()
    all_prices = []
    all_preds = []
    
    for ticker in tickers:
        # Fetch latest prices first (desc=True) then sort
        # Limit 5000 ensures ~20 years of history
        response = storage.client.table("stock_prices")\
            .select("*")\
            .eq("ticker", ticker)\
            .order("date", desc=True)\
            .limit(5000)\
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

# --- Sidebar ---
st.title("🔍 Stock peer analysis")
st.markdown("Easily compare stocks against others in their peer group.")

# Layout similar to image: Left column controls, Right column chart
col_controls, col_chart = st.columns([1, 3])

with col_controls:
    st.subheader("Stock tickers")
    # Multiselect with default peers
    selected_tickers = st.multiselect(
        "Select tickers",
        options=["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
        default=["NVDA", "AAPL", "MSFT"],
        label_visibility="collapsed"
    )
    
    st.subheader("Time horizon")
    # Pills for time range
    # Using radio as fallback if pills not available, but pills preferred
    try:
        time_range = st.pills(
            "Time horizon",
            options=["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "10 Years", "All"],
            default="1 Year",
            label_visibility="collapsed"
        )
    except AttributeError:
        time_range = st.radio(
            "Time horizon",
            options=["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "10 Years", "All"],
            index=3,
            horizontal=True,
            label_visibility="collapsed"
        )

    # Best/Worst Stock Logic (computed after loading data)
    st.markdown("---")
    
# --- Data Loading ---
if not selected_tickers:
    st.warning("Please select at least one ticker.")
else:
    with st.spinner("Loading data..."):
        df_prices, df_preds = load_data(selected_tickers)

    if df_prices.empty:
        st.error("No data found.")
    else:
        # Filter Date Range
        max_date = df_prices['date'].max()
        
        delta_map = {
            "1 Month": timedelta(days=30),
            "3 Months": timedelta(days=90),
            "6 Months": timedelta(days=180),
            "1 Year": timedelta(days=365),
            "5 Years": timedelta(days=365*5),
            "10 Years": timedelta(days=365*10),
            "All": None
        }
        
        if delta_map.get(time_range):
            start_date = max_date - delta_map[time_range]
        else:
            start_date = df_prices['date'].min()
            
        df_display = df_prices[df_prices['date'] >= start_date].copy()
        
        # Normalize
        df_display = normalize_prices(df_display, start_date)
        
        # Calculate Performance for Best/Worst
        perf_stats = []
        for t in selected_tickers:
            t_data = df_display[df_display['ticker'] == t]
            if not t_data.empty:
                start_price = t_data.iloc[0]['close']
                end_price = t_data.iloc[-1]['close']
                pct_change = (end_price - start_price) / start_price
                perf_stats.append({"ticker": t, "change": pct_change})
        
        if perf_stats:
            best_stock = max(perf_stats, key=lambda x: x['change'])
            worst_stock = min(perf_stats, key=lambda x: x['change'])
            
            with col_controls:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Best stock**")
                    st.metric(best_stock['ticker'], f"↑ {best_stock['change']*100:.0f}%", label_visibility="collapsed")
                with c2:
                    st.markdown("**Worst stock**")
                    st.metric(worst_stock['ticker'], f"↑ {worst_stock['change']*100:.0f}%" if worst_stock['change']>0 else f"↓ {abs(worst_stock['change'])*100:.0f}%", 
                              delta_color="normal" if worst_stock['change']>0 else "inverse", label_visibility="collapsed")

        # --- Chart ---
        with col_chart:
            fig = go.Figure()
            
            # Plot each ticker
            for t in selected_tickers:
                t_data = df_display[df_display['ticker'] == t]
                fig.add_trace(go.Scatter(
                    x=t_data['date'],
                    y=t_data['normalized_price'],
                    mode='lines',
                    name=t
                ))
                
            fig.update_layout(
                title="Normalized price",
                yaxis_title="Normalized price",
                xaxis_title="Date",
                height=600,
                hovermode="x unified",
                legend=dict(title="Stock"),
                template="plotly_dark" # Matches the dark theme request
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Prediction Section (Below Chart) ---
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
