from flows.daily_ingest import daily_ingest_flow

if __name__ == "__main__":
    # This will create a deployment and start a runner for it immediately
    # Run daily (every 24 hours = 86400 seconds)
    daily_ingest_flow.serve(
        name="daily-stock-prediction",
        interval=86400
    )