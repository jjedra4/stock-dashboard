from flows.daily_ingest import daily_ingest_flow

if __name__ == "__main__":
    # Runs daily
    daily_ingest_flow.serve(
        name="daily-stock-prediction",
        interval=86400
    )