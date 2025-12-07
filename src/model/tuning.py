import optuna
import wandb
import pandas as pd
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
                # "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                # "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
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

        # Ensure data is sorted by date
        if 'date' in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
                self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
            
            # Split train/test based on date
            split_date = pd.to_datetime("2025-06-01")
            
            # Check data range
            max_date = self.data['date'].max()
            if max_date < split_date:
                raise ValueError(f"Split date {split_date} is after the last date in data ({max_date}). Cannot create test set.")
            
            X_train = self.data[self.data['date'] < split_date].copy()
            X_test = self.data[self.data['date'] >= split_date].copy()
        else:
             # Fallback if no date column
             split_idx = int(len(self.data) * 0.8)
             X_train = self.data.iloc[:split_idx].copy()
             X_test = self.data.iloc[split_idx:].copy()
            
        # Create Model
        model = ModelFactory.get_model(self.model_type, **params)
        
        if self.model_type == "xgboost":
            model.train(X_train)
        else:
            pass
            
        def evaluate_split(X_split, split_name):
            try:
                # Add lookback context for prediction if needed (handled inside model or here)
                # But since we want to evaluate pure performance on X_split:
                preds = model.predict(X_split)
                
                price_col = 'adjClose' if 'adjClose' in X_split.columns else 'close'
                
                # Calculate actual returns for evaluation
                # Target is log return: ln(P_t+1 / P_t)
                actual_returns = np.log(X_split[price_col].shift(-1) / X_split[price_col])
                
                # Align preds and actuals
                valid_len = len(preds)
                actual_returns_aligned = actual_returns.iloc[-valid_len:]
                
                # Drop the last element (future unknown for actuals, prediction for t+1 for preds)
                # Ensure we have data
                if len(preds) <= 1:
                    return {}
                    
                preds_eval = preds[:-1]
                actual_eval = actual_returns_aligned.iloc[:-1]
                
                # --- METRICS ---
                
                # 1. RMSE (Root Mean Squared Error)
                rmse = np.sqrt(mean_squared_error(actual_eval, preds_eval))
                
                # 2. MAE (Mean Absolute Error)
                mae = mean_absolute_error(actual_eval, preds_eval)
                
                # 3. Directional Accuracy (DA)
                # Fraction of times the predicted sign matches the actual sign
                # We use np.sign. Note: sign(0) is 0.
                pred_dir = np.sign(preds_eval)
                actual_dir = np.sign(actual_eval)
                da = np.mean(pred_dir == actual_dir)
                
                # 4. Information Coefficient (IC) - Spearman Rank Correlation
                # Measures how well the model ranks returns (monotonicity)
                # preds_eval might be a Series or array. Ensure Series for corr.
                if not isinstance(preds_eval, pd.Series):
                    preds_series = pd.Series(preds_eval)
                else:
                    preds_series = preds_eval
                    
                # reset index to align for correlation if needed, or just values
                # actual_eval is a Series with index. preds_eval might have different index if created from np array
                ic = preds_series.corr(pd.Series(actual_eval.values), method='spearman')
                
                # 5. Sharpe Ratio (Annualized) of a simple Long/Short Strategy
                # Strategy: Long if pred > 0, Short if pred < 0
                # Returns = sign(pred) * actual_ret
                # Annualized factor = sqrt(252) for daily data
                strategy_rets = np.sign(preds_eval) * actual_eval
                sharpe = np.mean(strategy_rets) / (np.std(strategy_rets) + 1e-9) * np.sqrt(252)

                return {
                    f"{split_name}_rmse": rmse,
                    f"{split_name}_mae": mae,
                    f"{split_name}_da": da,
                    f"{split_name}_ic": ic,
                    f"{split_name}_sharpe": sharpe
                }
            except Exception as e:
                print(f"Error evaluating {split_name}: {e}")
                return {}

        train_metrics = evaluate_split(X_train, "train")
        test_metrics = evaluate_split(X_test, "test")
        
        # Log final metric
        all_metrics = {**train_metrics, **test_metrics}
        if "test_rmse" in all_metrics:
            all_metrics["rmse"] = all_metrics["test_rmse"]
            
        run.log(all_metrics)
        run.finish()
        
        return all_metrics.get("test_rmse", float('inf'))

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
    
    # TAIL IS THE NEWEST DATA!!!
    df = df.tail(2000)
    
    tuner = HyperparameterTuner(df, model_type="xgboost", n_trials=30)
    best_params = tuner.run()
