from ..interface import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostModel(BaseModel):
    def __init__(self, **params):
        self.params = params
        self.model = None  # To be trained

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        dtrain = self._prepare_data(X_train, y_train)
        
        # Separate n_estimators from other params
        train_params = self.params.copy()
        num_boost_round = train_params.pop("n_estimators", 100)
        
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if X.shape[0] < 30:
            raise ValueError("X must have at least 30 rows due to rolling windows")
        
        dtest = self._prepare_data(X)
        return self.model.predict(dtest)

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model = xgb.Booster()
        self.model.load_model(path)

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None) -> tuple[xgb.DMatrix, xgb.DMatrix]:
        df = X.copy()
        price_col = 'adjClose' if 'adjClose' in df.columns else 'close'
        # --- FEATURE ENGINEERING ---

        # A. Stationarity & Trends (Log Returns)
        # Replace raw price with daily log return: ln(P_t / P_t-1)
        df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))

        # B. Relative Volume
        # Compare current volume to 20-day average to normalize it
        df['vol_rel'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-9)

        # C. Price vs VWAP (Distance)
        # Percentage distance from VWAP
        df['dist_vwap'] = (df[price_col] - df['vwap']) / (df['vwap'] + 1e-9)

        # D. Intra-day Dynamics (Candlestick Shapes)
        # Body Size (normalized by open)
        df['body_size'] = (df['close'] - df['open']) / df['open']
        # Upper Shadow: High vs max(Open, Close)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        # Lower Shadow: Low vs min(Open, Close)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        # Close Location Value (Where inside the high-low range did it close?)
        denominator = df['high'] - df['low']
        # Avoid division by zero if High == Low
        df['clv'] = np.where(
            denominator == 0, 
            0, 
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / denominator
        )

        # E. Lag Features (Giving the model "Memory")
        # We lag the engineered features, NOT the raw prices
        lags = [1, 2, 3, 5, 10]
        for lag in lags:
            df[f'log_ret_lag_{lag}'] = df['log_ret'].shift(lag)
            df[f'vol_rel_lag_{lag}'] = df['vol_rel'].shift(lag)

        # F. Cyclical Time Encoding (Seasonality)
        # Assuming the index is a DatetimeIndex. If 'date' is a column, convert it first.
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find a date column if index isn't date
            for col in df.columns:
                if 'date' in str(col).lower():
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Day of week (0-6) converted to Sine/Cosine
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            # Month (1-12) converted to Sine/Cosine
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # --- CLEANUP ---
        
        # Drop raw non-stationary columns and metadata
        cols_to_drop = [
            'open', 'high', 'low', 'close', 'adjClose', 'volume', 
            'unadjustedVolume', 'change', 'changePercent', 'vwap', 
            'label', 'ticker', 'changeOverTime'
        ]
        # Only drop columns that actually exist in the dataframe
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Drop NaN values created by rolling windows and lags
        # (This will remove the first ~20 rows of data)
        df_clean = df.dropna()

        # --- TARGET PREPARATION ---
        
        if y is None:
            return xgb.DMatrix(df_clean)
        else:
            y_aligned = y.loc[df_clean.index]
            current_prices = X.loc[df_clean.index, price_col]

            y_transformed = np.log(y_aligned / current_prices)
            
            valid_mask = df_clean.notna().all(axis=1) & y_transformed.notna()
            
            # Apply mask to X and y so we don't pass any NaN values to the DMatrix
            X_clean = df_clean.loc[valid_mask]
            y_clean = y_transformed.loc[valid_mask]

            return xgb.DMatrix(X_clean, label=y_clean)
