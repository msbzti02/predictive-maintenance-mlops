import mlflow
from mlflow.tracking import MlflowClient
import yaml
import warnings
warnings.filterwarnings("ignore")


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    print("=" * 60)
    print("STAGE 5: MODEL REGISTRY AND GOVERNANCE")
    print("=" * 60)

    model_name = "PredictiveMaintenanceModel"
    experiment  = client.get_experiment_by_name(config["mlflow"]["experiment_name"])

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.RMSE ASC"],
        max_results=1,
    )
    best_run = runs[0]
    rmse     = best_run.data.metrics["RMSE"]
    run_id   = best_run.info.run_id

    print(f"Registering best run: {run_id}  (RMSE: {rmse:.2f})")

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    print(f"Model version {mv.version} registered.")

    if rmse < 35.0:
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Production"
        )
        print(f"Model promoted to PRODUCTION (RMSE {rmse:.2f} < 35).")
    else:
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Staging"
        )
        print(f"Model moved to STAGING (RMSE {rmse:.2f} did not meet threshold < 35).")

    print("Stage 5 completed.\n")


if __name__ == "__main__":
    main()
