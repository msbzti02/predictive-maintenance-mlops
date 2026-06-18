import pandas as pd
import shap
import mlflow
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from mlflow.tracking import MlflowClient


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    output_dir = config.get("output", {}).get("reports_dir", "outputs/reports")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STAGE 4: MODEL EXPLAINABILITY (SHAP)")
    print("=" * 60)

    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if not experiment:
        print("Experiment not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.RMSE ASC"],
        max_results=1,
    )
    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}  (RMSE: {best_run.data.metrics['RMSE']:.2f})")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model     = mlflow.sklearn.load_model(model_uri)

    train_df = pd.read_csv(config["data"]["train_path"])
    X_train  = train_df.drop(columns=config["data"]["drop_cols"] + [config["data"]["target_col"]])

    explainer   = shap.TreeExplainer(model)
    X_sample    = X_train.sample(100, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    shap_plot_path = os.path.join(output_dir, "shap_summary.png")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(shap_plot_path, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_id=best_run.info.run_id):
        mlflow.log_artifact(shap_plot_path)

    print(f"SHAP summary plot saved -> {shap_plot_path}")
    print("Stage 4 completed.\n")


if __name__ == "__main__":
    main()
