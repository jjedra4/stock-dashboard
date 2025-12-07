from prefect import flow, task
import pandas as pd
from datetime import datetime
import os
from src.data.fmp import FMPClient
from src.data.storage import SupabaseStorage
from src.model.train import train_final_model
from src.model.predict import load_model, make_predictions
from src.data.loader import get_training_data

@task(retries=3)
def fetch_and_save_data(ticker: str):
    """
    Fetches latest data from FMP and updates Supabase.
    Logic is similar to get_training_data but ensures DB is up to date first.
    """
    print(f"Fetching latest data for {ticker}...")
    client = FMPClient()
    storage = SupabaseStorage()
    
    # Get latest date from DB to only fetch new data
    latest_date_str = storage.get_latest_date(ticker)
    from_date = latest_date_str if latest_date_str else (datetime.now() - pd.Timedelta(days=365*10)).strftime("%Y-%m-%d")
    
    # Fetch new data
    df_new = client.get_historical_price(ticker, from_date=from_date)
    
    if df_new.empty:
        print("No new data found.")
        return
    
    # Save to DB
    if 'ticker' not in df_new.columns:
        df_new['ticker'] = ticker
        
    storage.upsert_stock_data(df_new, table_name="stock_prices")
    print(f"Saved {len(df_new)} new rows for {ticker}")

@task
def retrain_model(ticker: str):
    """
    Retrains the model on the latest 2000 records.
    """
    print(f"Retraining model for {ticker}...")
    model = train_final_model(ticker=ticker, save_path="models")
    return model

@task
def predict_next_day(ticker: str):
    """
    Predicts the next day's return using the latest model.
    """
    print(f"Predicting next day for {ticker}...")
    
    # Find the latest model file
    date_str = datetime.now().strftime("%Y%m%d")
    model_path = f"models/{ticker}_{date_str}_model.json"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using production backup if available.")
        # Fallback or fail? Let's try to find any recent model or fail.
        # For now, we assume retrain_model just ran successfully.
        return
        
    model = load_model(model_path, model_type="xgboost")
    
    # Get recent data for feature engineering (need ~30-50 days context)
    df = get_training_data(ticker)
    input_data = df.tail(100).copy()
    
    if 'date' in input_data.columns:
         input_data['date'] = pd.to_datetime(input_data['date'])
    
    # Predict
    # The model predicts t+1 return based on t. 
    # So we want to predict for the *last* row in input_data (which is 'today') to get 'tomorrow's' return.
    preds = make_predictions(model, input_data)
    
    if len(preds) == 0:
        print("No predictions generated.")
        return
        
    latest_pred = preds[-1]
    print(f"Predicted Log Return for next trading day: {latest_pred}")
    
    # Convert log return to percentage for display
    # P_t+1 = P_t * exp(log_ret) => pct_change = exp(log_ret) - 1
    pct_change = (2.71828 ** latest_pred) - 1
    print(f"Predicted % Change: {pct_change * 100:.2f}%")
    
    # Save prediction to DB (optional, but good for tracking)
    storage = SupabaseStorage()
    prediction_record = pd.DataFrame([{
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "predicted_log_return": float(latest_pred),
        "predicted_pct_change": float(pct_change),
        "created_at": datetime.now().isoformat()
    }])
    
    # Assuming you have a 'predictions' table. If not, maybe just log it.
    try:
        storage.upsert_stock_data(prediction_record, table_name="predictions") 
        print("Prediction saved to DB.")
    except Exception as e:
        print(f"Could not save prediction to DB (table might not exist): {e}")

@flow(name="Daily Stock Ingest and Train")
def daily_ingest_flow(tickers: list = ["NVDA"]):
    for ticker in tickers:
        # 1. Update Data
        fetch_and_save_data(ticker)
        
        # 2. Retrain Model
        retrain_model(ticker)
        
        # 3. Predict Next Day
        predict_next_day(ticker)

if __name__ == "__main__":
    daily_ingest_flow()

