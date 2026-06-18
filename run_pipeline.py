import subprocess
import sys
import datetime


def run_stage(script, name):
    print(f"\n{'=' * 60}")
    print(f"  RUNNING {name}")
    print(f"{'=' * 60}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n  FAILED: {name} -- Pipeline stopped.")
        sys.exit(1)


if __name__ == "__main__":
    started = datetime.datetime.now()
    print(f"\n{'=' * 60}")
    print(f"  MLOPS PIPELINE START  [{started.strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"{'=' * 60}")

    run_stage("src/data_preprocessing.py", "Stage 1: Data Preprocessing")
    run_stage("src/train.py",              "Stage 2: Multi-Model Training")
    run_stage("src/tune.py",               "Stage 3: Hyperparameter Tuning")
    run_stage("src/explain.py",            "Stage 4: SHAP Explainability")
    run_stage("src/register.py",           "Stage 5: Model Registry")
    run_stage("src/monitor.py",            "Stage 6: Performance Monitoring")

    finished = datetime.datetime.now()
    elapsed  = finished - started
    print(f"\n{'=' * 60}")
    print(f"  ALL STAGES COMPLETED SUCCESSFULLY")
    print(f"  Finished at : {finished.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time  : {str(elapsed).split('.')[0]}")
    print(f"{'=' * 60}")
    print("  Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view results.")
    print("  Run 'python src/serve.py' to start the prediction API.")
    print(f"{'=' * 60}\n")
