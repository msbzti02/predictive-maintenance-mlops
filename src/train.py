import pandas as pd
import numpy as np
import mlflow
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error, max_error
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlflow.models.signature import infer_signature


def phm08_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def eval_metrics(actual, pred):
    rmse    = np.sqrt(mean_squared_error(actual, pred))
    mae     = mean_absolute_error(actual, pred)
    mape    = mean_absolute_percentage_error(actual, pred)
    r2      = r2_score(actual, pred)
    max_err = max_error(actual, pred)
    phm08   = phm08_score(actual, pred)
    return rmse, mae, mape, r2, max_err, phm08


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    mlflow.enable_system_metrics_logging()

    output_dir = config.get("output", {}).get("plots_dir", "outputs/plots")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading preprocessed data...")
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df  = pd.read_csv(config["data"]["test_path"])

    drop_cols = config["data"]["drop_cols"]
    target    = config["data"]["target_col"]

    X_train = train_df.drop(columns=drop_cols + [target])
    y_train = train_df[target]
    X_test  = test_df.drop(columns=drop_cols + [target])
    y_test  = test_df[target]

    models = {
        "Dummy":           DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge":           Ridge(),
        "RandomForest":    RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost":         XGBRegressor(n_estimators=50, random_state=42),
        "LightGBM":        LGBMRegressor(n_estimators=50, random_state=42),
        "CatBoost":        CatBoostRegressor(iterations=50, verbose=0, random_state=42),
    }

    print("=" * 60)
    print("STAGE 2: MULTI-MODEL TRAINING AND TRACKING")
    print("=" * 60)

    for name, model in models.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tags({
                "developer":       "Zubir",
                "project":         "Predictive_Maintenance_NASA",
                "algorithm":       name,
                "environment":     "training",
                "dataset_version": "v1.0",
            })
            mlflow.log_params({
                "train_rows":     len(X_train),
                "train_features": len(X_train.columns),
                "model_type":     name,
            })

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse, mae, mape, r2, max_err, phm08 = eval_metrics(y_test, preds)
            mlflow.log_metrics({
                "RMSE":       rmse,
                "MAE":        mae,
                "MAPE":       mape,
                "R2":         r2,
                "Max_Error":  max_err,
                "PHM08_Score": phm08,
            })

            plot_path = os.path.join(output_dir, f"accuracy_plot_{name}.png")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, preds, alpha=0.3, color="#00d2ff")
            plt.plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()], "r--", linewidth=2)
            plt.xlabel("Actual RUL")
            plt.ylabel("Predicted RUL")
            plt.title(f"Prediction Accuracy: {name}", fontweight="bold")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(plot_path, bbox_inches="tight", dpi=150)
            mlflow.log_artifact(plot_path)
            plt.close()

            signature = infer_signature(X_train, preds)
            mlflow.sklearn.log_model(model, "model", signature=signature)
            print(f"[{name:<16}]  RMSE: {rmse:>7.2f}  MAPE: {mape:>5.2f}  PHM08: {phm08:>14.2f}")


if __name__ == "__main__":
    main()
