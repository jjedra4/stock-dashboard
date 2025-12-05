import optuna
import wandb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model.factory import ModelFactory
import numpy as np
from dotenv import load_dotenv

class HyperparameterTuner:
    def __init__(self, data: pd.DataFrame, model_type: str, target_col: str = 'close', n_trials: int = 20):
        self.data = data
        self.model_type = model_type
        self.target_col = target_col
        self.n_trials = n_trials
        
    def _get_search_space(self, trial):
        """
        Defines hyperparameter search space for each model type.
        """
        if self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 16),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            }
        # Add more models later
        return {}

    def objective(self, trial):
        params = self._get_search_space(trial)
        
        # Initialize W&B run for this trial
        run = wandb.init(
            project="stock-dashboard", 
            group=f"tuning-{self.model_type}",
            config=params,
            reinit=True,
            job_type="optimization"
        )

        tscv = TimeSeriesSplit(n_splits=4)
        scores = []
        
        # Ensure data is sorted by date
        if 'date' in self.data.columns:
            self.data = self.data.sort_values('date')
        
        X, y = self.data, self.data[self.target_col]
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create Model
            model = ModelFactory.get_model(self.model_type, **params)
            
            if self.model_type == "xgboost":
                model.train(X_train)
            else:
                pass
            # Handle other models later
            
            preds = model.predict(X_val)

            price_col = 'adjClose' if 'adjClose' in X_val.columns else 'close'
            
            # Align y_val/X_val with preds (which are shorter due to lags)
            valid_len = len(preds)
            X_val_aligned = X_val.iloc[-valid_len:]
            y_val_aligned = y_val.iloc[-valid_len:]
            
            current_prices = X_val_aligned[price_col]
            y_val_log_ret = np.log(y_val_aligned / current_prices)
            
            rmse = np.sqrt(mean_squared_error(y_val_log_ret, preds))
            scores.append(rmse)
            
            # Optional: Log fold metric
            run.log({f"fold_{fold}_rmse": rmse})
            
        avg_score = np.mean(scores)
        
        # Log final metric
        run.log({"rmse": avg_score})
        run.finish()
        
        return avg_score

    def run(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print("Best params:", study.best_params)
        print("Best RMSE:", study.best_value)
        return study.best_params

if __name__ == "__main__":
    from src.data.loader import get_training_data
    
    load_dotenv()
    df = get_training_data("NVDA")
    
    tuner = HyperparameterTuner(df, model_type="xgboost", n_trials=100)
    best_params = tuner.run()

