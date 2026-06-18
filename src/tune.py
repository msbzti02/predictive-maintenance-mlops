import optuna
import pandas as pd
import numpy as np
import mlflow
import yaml
import logging
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    train_df = pd.read_csv(config["data"]["train_path"])
    test_df  = pd.read_csv(config["data"]["test_path"])

    X_train = train_df.drop(columns=config["data"]["drop_cols"] + [config["data"]["target_col"]])
    y_train = train_df[config["data"]["target_col"]]
    X_test  = test_df.drop(columns=config["data"]["drop_cols"] + [config["data"]["target_col"]])
    y_test  = test_df[config["data"]["target_col"]]

    print("=" * 60)
    print("STAGE 3: HYPERPARAMETER TUNING (Optuna + LightGBM)")
    print("=" * 60)

    best_rmse = [float("inf")]

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators",  50, 300),
            "max_depth":     trial.suggest_int("max_depth",      3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves",     20, 150),
            "random_state":  42,
            "verbose":       -1,
        }
        with mlflow.start_run(run_name="optuna_trial", nested=True):
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse  = np.sqrt(mean_squared_error(y_test, preds))
            mlflow.log_params(params)
            mlflow.log_metric("RMSE", rmse)
            if rmse < best_rmse[0]:
                best_rmse[0] = rmse
                mlflow.sklearn.log_model(model, "model")
        return rmse

    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config["optuna"]["n_trials"])
        print(f"\nTuning complete.  Best RMSE: {study.best_value:.2f}")
        print(f"Best params: {study.best_params}")
        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})


if __name__ == "__main__":
    main()
