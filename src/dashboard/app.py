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
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data(ticker):
    storage = SupabaseStorage()
    # Fetch price history
    # We need a method to fetch all or range. Using internal client for now if loader doesn't support range query efficiently
    # Or better, use the storage client directly
    response = storage.client.table("stock_prices").select("*").eq("ticker", ticker).order("date", desc=False).execute()
    df_prices = pd.DataFrame(response.data)
    
    if not df_prices.empty:
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        
    # Fetch predictions
    response_pred = storage.client.table("predictions").select("*").eq("ticker", ticker).order("date", desc=False).execute()
    df_preds = pd.DataFrame(response_pred.data)
    
    if not df_preds.empty:
        df_preds['date'] = pd.to_datetime(df_preds['date'])
        
    return df_prices, df_preds

def calculate_metrics(df_prices, df_preds):
    if df_prices.empty or df_preds.empty:
        return {}
    
    # Merge on date to compare
    # Prediction on date T is for return at T+1 usually, or T?
    # In daily_ingest.py: date=now(), predicted_log_return=next_day_return
    # So prediction made on 2023-01-01 is for close of 2023-01-02 (approx)
    # Let's align: Prediction Date is the date it was MADE.
    # We want to compare Prediction(T) with Actual Return (T -> T+1)
    
    merged = pd.merge(df_preds, df_prices, on="ticker", suffixes=('_pred', '_actual'))
    # We need to match prediction date with the price date
    # Actually, let's look at how we save:
    # date = datetime.now() -> This is "today"
    # predicted_log_return -> "tomorrow" return
    
    # So we join df_preds['date'] with df_prices['date'] to get Price_T
    merged = pd.merge(df_preds, df_prices, left_on="date", right_on="date", how="inner")
    
    # Now we need Price_T+1 to calculate actual return
    merged['next_close'] = merged['close'].shift(-1) # This assumes merged is sorted by date?
    # Wait, we can't shift on merged if it has gaps. Better to use full price history for shifts.
    
    # Better approach: Calculate actual returns in df_prices first
    df_prices = df_prices.sort_values('date')
    df_prices['actual_log_ret'] = np.log(df_prices['close'].shift(-1) / df_prices['close'])
    df_prices['actual_pct_change'] = df_prices['close'].shift(-1) / df_prices['close'] - 1
    
    # Now merge prediction with this
    comparison = pd.merge(df_preds, df_prices[['date', 'actual_log_ret', 'actual_pct_change', 'close']], on='date', how='inner')
    comparison = comparison.dropna()
    
    if comparison.empty:
        return {}
        
    # Metrics
    mae = np.mean(np.abs(comparison['predicted_log_return'] - comparison['actual_log_ret']))
    rmse = np.sqrt(np.mean((comparison['predicted_log_return'] - comparison['actual_log_ret'])**2))
    
    # Directional Accuracy
    pred_dir = np.sign(comparison['predicted_log_return'])
    actual_dir = np.sign(comparison['actual_log_ret'])
    da = np.mean(pred_dir == actual_dir)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Directional Accuracy": da,
        "Count": len(comparison)
    }

# --- Sidebar ---
st.sidebar.title("Settings")
ticker = st.sidebar.selectbox("Select Ticker", ["NVDA", "AAPL", "TSLA", "MSFT"], index=0)

time_range = st.sidebar.selectbox("Time Range", ["1 Week", "1 Month", "3 Months", "1 Year", "All"], index=3)

# --- Main Content ---
st.title(f"📈 {ticker} Stock Dashboard")

# Load Data
with st.spinner("Loading data..."):
    df_prices, df_preds = load_data(ticker)

if df_prices.empty:
    st.error(f"No data found for {ticker}")
else:
    # Filter by Date
    max_date = df_prices['date'].max()
    if time_range == "1 Week":
        start_date = max_date - timedelta(weeks=1)
    elif time_range == "1 Month":
        start_date = max_date - timedelta(days=30)
    elif time_range == "3 Months":
        start_date = max_date - timedelta(days=90)
    elif time_range == "1 Year":
        start_date = max_date - timedelta(days=365)
    else:
        start_date = df_prices['date'].min()
        
    df_display = df_prices[df_prices['date'] >= start_date]
    
    # --- Chart ---
    st.subheader("Price History & Predictions")
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_display['date'],
        open=df_display['open'],
        high=df_display['high'],
        low=df_display['low'],
        close=df_display['close'],
        name="Price"
    ))
    
    # Predictions Overlay (Arrows? or Dots?)
    # Let's plot recent predictions. 
    # If we have a prediction on date T, it predicts movement for T+1.
    # Let's calculate "Predicted Price" = Price_T * exp(pred_ret)
    if not df_preds.empty:
        pred_display = df_preds[df_preds['date'] >= start_date]
        merged_pred = pd.merge(pred_display, df_display[['date', 'close']], on='date', how='inner')
        
        if not merged_pred.empty:
            merged_pred['predicted_next_close'] = merged_pred['close'] * np.exp(merged_pred['predicted_log_return'])
            # Shift x axis by 1 day for visualization? 
            # Actually, let's just plot a marker at T indicating UP/DOWN or the target price
            
            # Up arrows
            up_preds = merged_pred[merged_pred['predicted_log_return'] > 0]
            fig.add_trace(go.Scatter(
                x=up_preds['date'], 
                y=up_preds['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name="Predict Up"
            ))
            
            # Down arrows
            down_preds = merged_pred[merged_pred['predicted_log_return'] < 0]
            fig.add_trace(go.Scatter(
                x=down_preds['date'], 
                y=down_preds['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name="Predict Down"
            ))

    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Latest Prediction ---
    st.subheader("Latest Prediction")
    if not df_preds.empty:
        latest_pred = df_preds.iloc[-1]
        pred_date = latest_pred['date']
        
        # Check if it's fresh (made today or yesterday)
        is_fresh = (datetime.now() - pred_date).days <= 3 # Allow weekend gaps
        
    col1, col2, col3 = st.columns(3)
        col1.metric("Prediction Date", pred_date.strftime("%Y-%m-%d"))
        col2.metric("Predicted Return", f"{latest_pred['predicted_pct_change']*100:.2f}%", delta_color="normal")
        
        direction = "Bullish 🟢" if latest_pred['predicted_log_return'] > 0 else "Bearish 🔴"
        col3.metric("Signal", direction)
        
        if not is_fresh:
            st.warning("⚠️ Latest prediction is old. Model might not be running.")
else:
        st.info("No predictions available yet.")

    # --- Model Stats ---
    st.subheader("Model Performance (Backtest)")
    metrics = calculate_metrics(df_prices, df_preds)
    
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Directional Accuracy", f"{metrics['Directional Accuracy']*100:.1f}%")
        m2.metric("RMSE (Log Ret)", f"{metrics['RMSE']:.5f}")
        m3.metric("MAE (Log Ret)", f"{metrics['MAE']:.5f}")
        m4.metric("Evaluated Predictions", metrics['Count'])
        
        # Detailed Table
        with st.expander("View Prediction History"):
            # Re-create comparison df for display
            df_prices_sorted = df_prices.sort_values('date')
            df_prices_sorted['actual_log_ret'] = np.log(df_prices_sorted['close'].shift(-1) / df_prices_sorted['close'])
            comparison = pd.merge(df_preds, df_prices_sorted[['date', 'actual_log_ret', 'close']], on='date', how='inner')
            comparison['error'] = comparison['predicted_log_return'] - comparison['actual_log_ret']
            comparison['correct_dir'] = np.sign(comparison['predicted_log_return']) == np.sign(comparison['actual_log_ret'])
            
            st.dataframe(comparison.sort_values('date', ascending=False))
    else:
        st.info("Not enough data to calculate accuracy (need actuals for next day).")
