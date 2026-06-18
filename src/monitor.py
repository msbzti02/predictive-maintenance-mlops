import pandas as pd
import mlflow
import yaml
import os
import warnings
warnings.filterwarnings("ignore")
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    output_dir = config.get("output", {}).get("reports_dir", "outputs/reports")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STAGE 6: PERFORMANCE MONITORING (Evidently AI)")
    print("=" * 60)

    print("Analyzing sensor data for drift...")
    reference = pd.read_csv(config["data"]["train_path"]).drop(columns=config["data"]["drop_cols"])
    current   = pd.read_csv(config["data"]["test_path"]).drop(columns=config["data"]["drop_cols"])

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(report_path)

    with mlflow.start_run(run_name="Evidently_Drift_Monitoring"):
        mlflow.log_artifact(report_path)

    print(f"Drift report saved -> {report_path}")
    print("Stage 6 completed.\n")


if __name__ == "__main__":
    main()
