# Predictive Maintenance MLOps Pipeline
## NASA C-MAPSS Turbofan Engine — Remaining Useful Life (RUL) Prediction

**Author:** Zubir  
**Date:** 2026-05-15  
**Course:** MLOps — Term Project  
**Dataset:** NASA C-MAPSS FD001  
**Experiment:** `Predictive_Maintenance_RUL`  
**Tracking Backend:** MLflow + SQLite  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [MLOps Architecture](#2-mlops-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Stage 1 — Data Ingestion & Preprocessing](#4-stage-1--data-ingestion--preprocessing)
5. [Stage 2 — Multi-Model Training & Experiment Tracking](#5-stage-2--multi-model-training--experiment-tracking)
6. [Stage 3 — Hyperparameter Tuning with Optuna](#6-stage-3--hyperparameter-tuning-with-optuna)
7. [Stage 4 — Model Explainability (SHAP)](#7-stage-4--model-explainability-shap)
8. [Stage 5 — Model Registry & Governance](#8-stage-5--model-registry--governance)
9. [Stage 6 — Data Drift & Performance Monitoring](#9-stage-6--data-drift--performance-monitoring)
10. [Pipeline Execution Log](#10-pipeline-execution-log)
11. [MLflow Dashboard](#11-mlflow-dashboard)
12. [Containerization](#12-containerization)
13. [Prediction API](#13-prediction-api)
14. [Conclusions](#14-conclusions)

---

## 1. Project Overview

This project implements a **production-grade, end-to-end MLOps pipeline** for predicting the Remaining Useful Life (RUL) of turbofan aircraft engines using the NASA C-MAPSS dataset.

The pipeline follows the **MLOps Maturity Model Level 2** — fully automated training, experiment tracking, model versioning, explainability, and drift monitoring, all governed by MLflow.

### Problem Statement

> Given multivariate time-series sensor readings from a turbofan engine, predict the number of operational cycles remaining before failure (RUL). Early and accurate RUL prediction enables proactive maintenance scheduling, reducing unplanned downtime and safety incidents.

### Key MLOps Goals

| Goal | Implementation |
|------|---------------|
| Reproducibility | Config-driven pipeline (`configs/config.yaml`) |
| Experiment Tracking | MLflow with SQLite backend |
| Automated Governance | Model Registry with stage transitions |
| Interpretability | SHAP TreeExplainer on best model |
| Monitoring | Evidently AI data & target drift reports |
| Serving | FastAPI REST prediction endpoint |
| Containerization | Docker + docker-compose |

---

## 2. MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    run_pipeline.py                          │
│              (Orchestrator — Sequential Stages)             │
└──────────┬──────────────────────────────────────────────────┘
           │
     ┌─────▼──────┐    ┌──────────────┐    ┌──────────────┐
     │  Stage 1   │    │   Stage 2    │    │   Stage 3    │
     │   Data     │───▶│  Multi-Model │───▶│   Optuna     │
     │  Ingest &  │    │  Training    │    │   Tuning     │
     │   Preproc  │    │  (7 models)  │    │  (10 trials) │
     └────────────┘    └──────┬───────┘    └──────┬───────┘
                              │                   │
                        ┌─────▼───────────────────▼──────┐
                        │         MLflow Tracking         │
                        │    (Params, Metrics, Artifacts) │
                        └─────────────┬──────────────────┘
                                      │
     ┌────────────┐    ┌──────────────▼──┐    ┌──────────────┐
     │  Stage 6   │    │    Stage 4      │    │   Stage 5    │
     │ Evidently  │◀───│  SHAP Explain-  │───▶│   Model      │
     │ Monitoring │    │  ability        │    │   Registry   │
     └────────────┘    └─────────────────┘    └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │  FastAPI     │
                                              │  Serve.py    │
                                              │  /predict    │
                                              └──────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Python subprocess pipeline |
| Experiment Tracking | MLflow 2.12.1 + SQLite |
| ML Models | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Hyperparameter Tuning | Optuna 3.6.1 |
| Explainability | SHAP 0.45.0 |
| Drift Monitoring | Evidently AI 0.4.19 |
| API Serving | FastAPI + Uvicorn |
| Containerization | Docker + docker-compose |

---

## 3. Repository Structure

```
Term_project_zubir/
├── configs/
│   └── config.yaml              # Central pipeline configuration
├── data/
│   ├── raw/                     # NASA C-MAPSS FD001 text files
│   └── processed/               # Feature-engineered CSV files
├── docs/
│   └── MLOPS_REPORT.md          # This report
├── outputs/
│   ├── plots/                   # Per-model accuracy scatter plots
│   └── reports/                 # SHAP summary + Evidently HTML
├── src/
│   ├── data_preprocessing.py    # Stage 1
│   ├── train.py                 # Stage 2
│   ├── tune.py                  # Stage 3
│   ├── explain.py               # Stage 4
│   ├── register.py              # Stage 5
│   ├── monitor.py               # Stage 6
│   └── serve.py                 # FastAPI prediction server
├── tests/
│   └── test_pipeline.py
├── mlruns/                      # MLflow artifact store
├── mlflow.db                    # MLflow SQLite tracking backend
├── run_pipeline.py              # Master pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 4. Stage 1 — Data Ingestion & Preprocessing

### Overview

The NASA C-MAPSS FD001 dataset contains multivariate time-series sensor readings from 100 turbofan engines operating under a single flight condition. Each engine starts healthy and degrades to failure.

**Raw Data Columns:** `engine_id`, `cycle`, `setting1-3`, `s1-s21` (21 sensors)

### RUL Calculation

For **training data**: RUL at each cycle = `max_cycle(engine) − current_cycle`

For **test data**: RUL is anchored using the ground-truth `RUL_FD001.txt` file, then back-calculated for all prior cycles.

### Feature Engineering

Six zero-variance sensors (`s1, s5, s10, s16, s18, s19`) are dropped. For each remaining sensor, two rolling window features are generated per engine:

- **5-cycle rolling mean** — captures degradation trend
- **5-cycle rolling std** — captures degradation volatility

| Dataset | Raw Shape | Processed Shape |
|---------|-----------|-----------------|
| Train | (20,631 × 26) | (20,631 × 51) |
| Test | (13,096 × 26) | (13,096 × 51) |

**Features added:** 30 rolling features (15 sensors × 2 stats)

### Evidence — Stage 1 Terminal Output

```
============================================================
STAGE 1: DATA INGESTION AND PREPROCESSING
============================================================
Loading raw text files from data/raw...
Raw Train Shape: (20631, 26)
Raw Test Shape:  (13096, 26)
Applying time-series feature engineering...
Applying time-series feature engineering...
Processed Train saved -> data/processed\train.csv | Shape: (20631, 51)
Processed Test saved  -> data/processed\test.csv  | Shape: (13096, 51)
Stage 1 completed.
```

![Stage 1 — Terminal output showing raw → processed shapes](docs/screenshots/stage1_terminal.png)


---

## 5. Stage 2 — Multi-Model Training & Experiment Tracking

### Overview

Seven regression models are trained in a single automated sweep. Every run is logged to MLflow with full parameter, metric, artifact, and model signature tracking — enabling complete experiment governance.

### MLflow Run Configuration

Each run logs:
- **Tags:** `developer`, `project`, `algorithm`, `environment`, `dataset_version`
- **Params:** `train_rows`, `train_features`, `model_type`
- **Metrics:** 6 metrics (see below)
- **Artifacts:** accuracy scatter plot PNG
- **Model:** serialized model + input/output signature
- **System metrics:** CPU & RAM usage during training

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — primary ranking metric |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |
| **R²** | Coefficient of Determination |
| **Max Error** | Worst single prediction error |
| **PHM08 Score** | Asymmetric penalty score from the PHM08 competition |

**PHM08 Score formula:**

```
d = y_pred - y_true
score = sum( exp(-d/13)-1  if d < 0  else  exp(d/10)-1 )
```

Late predictions (positive d) are penalized more heavily than early ones — reflecting real-world maintenance cost asymmetry.

### Model Leaderboard

| Rank | Model | RMSE | MAE | MAPE | R² | PHM08 Score |
|------|-------|------|-----|------|-----|-------------|
| 1 | **LightGBM** | **46.12** | — | 0.28 | — | 122,456,717 |
| 2 | RandomForest | 47.08 | — | 0.29 | — | 152,114,063 |
| 3 | Ridge | 47.74 | — | 0.32 | — | 218,779,371 |
| 4 | LinearRegression | 47.80 | — | 0.32 | — | 238,656,082 |
| 5 | CatBoost | 48.43 | — | 0.30 | — | 163,061,461 |
| 6 | XGBoost | 48.75 | — | 0.30 | — | 214,521,685 |
| 7 | Dummy (baseline) | 67.79 | — | 0.49 | — | 1,441,084,108 |

### Evidence — Stage 2 Terminal Output

```
============================================================
STAGE 2: MULTI-MODEL TRAINING AND TRACKING
============================================================
[Dummy           ]  RMSE:   67.79  MAPE:  0.49  PHM08:  1441084108.66
[LinearRegression]  RMSE:   47.80  MAPE:  0.32  PHM08:   238656082.45
[Ridge           ]  RMSE:   47.74  MAPE:  0.32  PHM08:   218779371.65
[RandomForest    ]  RMSE:   47.08  MAPE:  0.29  PHM08:   152114063.83
[XGBoost         ]  RMSE:   48.75  MAPE:  0.30  PHM08:   214521685.04
[LightGBM        ]  RMSE:   46.12  MAPE:  0.28  PHM08:   122456717.77
[CatBoost        ]  RMSE:   48.43  MAPE:  0.30  PHM08:   163061461.62
```

![Stage 2: MLflow Experiments UI showing all 7 runs](docs/screenshots/stage2_mlflow_runs.png)


![Stage 2: MLflow metric comparison chart (RMSE across models)](docs/screenshots/stage2_rmse_comparison.png)


![Stage 2: LightGBM accuracy scatter plot (best model)](outputs/plots/accuracy_plot_LightGBM.png)


---

## 6. Stage 3 — Hyperparameter Tuning with Optuna

### Overview

**Optuna** performs automated black-box hyperparameter optimization on the best model class (LightGBM) using Tree-structured Parzen Estimator (TPE) sampling over 10 trials. Every trial is logged as a **nested MLflow run** under a parent `Hyperparameter_Tuning` run.

### Search Space

| Hyperparameter | Type | Range |
|---------------|------|-------|
| `n_estimators` | int | 50 – 300 |
| `max_depth` | int | 3 – 10 |
| `learning_rate` | float (log) | 0.01 – 0.30 |
| `num_leaves` | int | 20 – 150 |

### Results

| | Value |
|--|-------|
| **Best RMSE** | **45.18** |
| **Best n_estimators** | 217 |
| **Best max_depth** | 3 |
| **Best learning_rate** | 0.0328 |
| **Best num_leaves** | 47 |
| Improvement over baseline LightGBM | −0.94 RMSE |

### Evidence — Stage 3 Terminal Output

```
============================================================
STAGE 3: HYPERPARAMETER TUNING (Optuna + LightGBM)
============================================================

Tuning complete.  Best RMSE: 45.18
Best params: {'n_estimators': 217, 'max_depth': 3,
              'learning_rate': 0.032819998926021415, 'num_leaves': 47}
```

![Stage 3: MLflow nested runs showing all 10 Optuna trials](docs/screenshots/stage3_optuna_trials.png)

![Stage 3: Optuna optimization history plot (RMSE vs trial)](docs/screenshots/stage3_optuna_history.png)

---

## 7. Stage 4 — Model Explainability (SHAP)

### Overview

**SHAP (SHapley Additive exPlanations)** is applied to the best-performing model (tuned LightGBM, RMSE: 45.18) to provide global feature importance analysis. This satisfies the MLOps requirement for **model interpretability and auditability**.

- Explainer: `shap.TreeExplainer` (exact, fast for tree models)
- Sample: 100 random training observations
- Output: SHAP summary plot (beeswarm) logged to MLflow artifacts

### What SHAP Tells Us

The summary plot reveals which sensors drive RUL predictions most:
- Features with high absolute SHAP values have the greatest influence
- Red = high feature value, Blue = low feature value
- Position on x-axis shows direction of impact (positive = increases predicted RUL)

### Evidence — SHAP Summary Plot

![Stage 4: SHAP summary beeswarm plot](outputs/reports/shap_summary.png)


![Stage 4: MLflow artifact page showing shap_summary.png logged](docs/screenshots/stage4_mlflow_artifact.png)

### Evidence — Stage 4 Terminal Output

```
============================================================
STAGE 4: MODEL EXPLAINABILITY (SHAP)
============================================================
Best run ID: a02f687a4aec493f96a9844df2fcb295  (RMSE: 45.18)
SHAP summary plot saved -> outputs/reports\shap_summary.png
Stage 4 completed.
```

---

## 8. Stage 5 — Model Registry & Governance

### Overview

The best model is promoted through the **MLflow Model Registry**, implementing a structured model lifecycle with stage gates. This is a core **MLOps governance** practice.

### Registry Workflow

```
[Candidate Run] ──► [Register] ──► [Staging] ──► [Production]
                                      ▲                ▲
                                  RMSE ≥ 35        RMSE < 35
                                  (current)       (threshold)
```

### Stage Gate Policy

| Condition | Action |
|-----------|--------|
| RMSE < 35.0 | Promote to **Production** |
| RMSE ≥ 35.0 | Hold in **Staging** |

### Current Status

| Field | Value |
|-------|-------|
| Model Name | `PredictiveMaintenanceModel` |
| Version | 2 |
| Stage | **Staging** |
| Best RMSE | 45.18 |
| Registered Run | `a02f687a4aec493f96a9844df2fcb295` |
| Model URI | `models:/PredictiveMaintenanceModel/Staging` |

> The model is held in **Staging** as RMSE (45.18) exceeds the Production gate threshold of 35.0 — a domain-standard target for the FD001 dataset. Further tuning with deeper feature engineering or ensemble stacking is required for Production promotion.

### Evidence — Stage 5 Terminal Output

```
============================================================
STAGE 5: MODEL REGISTRY AND GOVERNANCE
============================================================
Registering best run: a02f687a4aec493f96a9844df2fcb295  (RMSE: 45.18)
Model version 2 registered.
Model moved to STAGING (RMSE 45.18 did not meet threshold < 35).
Stage 5 completed.
```

![Stage 5: MLflow Model Registry UI showing PredictiveMaintenanceModel](docs/screenshots/stage5_model_registry.png)

![Stage 5: Model version details page with Staging badge](docs/screenshots/stage5_model_version.png)


---

## 9. Stage 6 — Data Drift & Performance Monitoring

### Overview

**Evidently AI** is used to simulate production monitoring by comparing the training dataset (reference) against the test dataset (current/production). This detects whether the data distribution has shifted — a critical signal for model retraining triggers in MLOps.

### Metrics Generated

| Report | Description |
|--------|-------------|
| **DataDriftPreset** | Per-feature statistical drift tests (KS test, PSI) |
| **TargetDriftPreset** | Distribution shift in the RUL target variable |

### Drift Report Artifacts

- `outputs/reports/drift_report.html` — full interactive HTML dashboard
- Logged to MLflow under run: `Evidently_Drift_Monitoring`

### Evidence — Stage 6 Terminal Output

```
============================================================
STAGE 6: PERFORMANCE MONITORING (Evidently AI)
============================================================
Analyzing sensor data for drift...
Drift report saved -> outputs/reports\drift_report.html
Stage 6 completed.
```

![Stage 6: Evidently drift report HTML dashboard (overview)](docs/screenshots/stage6_drift_overview.png)


![Stage 6: Evidently per-feature drift detail (top drifted sensors)](docs/screenshots/stage6_drift_detail.png)


![Stage 6: MLflow artifact page showing drift_report.html logged](docs/screenshots/stage6_mlflow_artifact.png)


---

## 10. Pipeline Execution Log

The full pipeline is orchestrated by `run_pipeline.py` — a single command that runs all 6 stages sequentially with fail-fast error handling.

### Execution Command

```bash
python run_pipeline.py
```

### Full Run Log

```
============================================================
  MLOPS PIPELINE START  [2026-05-15 23:52:04]
============================================================

============================================================
  RUNNING Stage 1: Data Preprocessing
============================================================
[Stage 1 output ...]
Stage 1 completed.

============================================================
  RUNNING Stage 2: Multi-Model Training
============================================================
[Stage 2 output — 7 models logged ...]

============================================================
  RUNNING Stage 3: Hyperparameter Tuning
============================================================
Tuning complete.  Best RMSE: 45.18

============================================================
  RUNNING Stage 4: SHAP Explainability
============================================================
Stage 4 completed.

============================================================
  RUNNING Stage 5: Model Registry
============================================================
Model version 2 registered.
Stage 5 completed.

============================================================
  RUNNING Stage 6: Performance Monitoring
============================================================
Stage 6 completed.

============================================================
  ALL STAGES COMPLETED SUCCESSFULLY
  Finished at : 2026-05-15 23:54:15
  Total time  : 0:02:10
============================================================
```

![Full pipeline terminal output (complete run log)](docs/screenshots/pipeline_full_run.png)


---

## 11. MLflow Dashboard

MLflow provides the central experiment tracking UI accessible at `http://localhost:5000`.

### Experiment View

![MLflow: Experiment list showing Predictive_Maintenance_RUL](docs/screenshots/mlflow_experiment_list.png)

### Runs Comparison Table

![MLflow: All runs table sorted by RMSE ascending](docs/screenshots/mlflow_runs_table.png)


### Best Run Detail Page

![MLflow: Best run (a02f687a) detail — params, metrics, artifacts](docs/screenshots/mlflow_best_run_detail.png)


### System Metrics (CPU/RAM)

MLflow system metrics logging is enabled during training (`mlflow.enable_system_metrics_logging()`), capturing real-time CPU and RAM usage per run.

![MLflow: System metrics tab showing CPU/RAM during training](docs/screenshots/mlflow_system_metrics.png)


### Launch Command

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## 12. Containerization

The project is fully containerized using Docker for environment reproducibility.

### Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "src/serve.py"]
```

### docker-compose.yml Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI prediction server |
| `mlflow` | 5000 | MLflow tracking UI |

### Run with Docker Compose

```bash
docker-compose up --build
```

![Docker: docker-compose up output showing both services running](docs/screenshots/docker_compose_up.png)


---

## 13. Prediction API

A **FastAPI** REST server exposes the registered model for real-time RUL inference.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | RUL prediction |
| GET | `/docs` | Swagger UI |

### Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"s2": 641.82, "s3": 1589.7, "s4": 1400.6, "s6": 21.61}}'
```

### Sample Response

```json
{
  "predicted_RUL_cycles": 87.43,
  "maintenance_recommendation": "Healthy"
}
```

### Start API

```bash
python src/serve.py
```

![FastAPI: Swagger UI at /docs showing /predict endpoint](docs/screenshots/api_swagger_ui.png)


![FastAPI: Sample /predict POST request and JSON response](docs/screenshots/api_predict_response.png)


---

## 14. Conclusions

### Pipeline Summary

| Item | Result |
|------|--------|
| Total Stages | 6 |
| Total Runtime | 2 min 10 sec |
| Models Evaluated | 7 |
| Best Model | LightGBM (tuned, Optuna) |
| Best RMSE | **45.18** |
| Model Registry Stage | Staging |
| Explainability | SHAP TreeExplainer |
| Drift Monitoring | Evidently AI |
| API Status | FastAPI — Ready |

### MLOps Maturity Assessment

| Dimension | Level | Notes |
|-----------|-------|-------|
| Reproducibility | ✅ High | Config-driven, versioned data |
| Experiment Tracking | ✅ High | Full MLflow logging — params, metrics, artifacts, tags |
| Model Versioning | ✅ High | MLflow Model Registry with stage gates |
| Automation | ✅ High | Single-command `run_pipeline.py` orchestration |
| Interpretability | ✅ High | SHAP global feature importance |
| Monitoring | ✅ High | Evidently data & target drift |
| CI/CD | Partial | Manual trigger; can be extended with GitHub Actions |
| Serving | ✅ High | FastAPI + Docker |

### Future Work

- **Lower RMSE below 35** to achieve Production promotion — via LSTM/CNN time-series models
- **Piecewise linear RUL capping** (RUL max = 125) to reduce noise in early cycles
- **Automated retraining trigger** when drift score exceeds threshold
- **CI/CD integration** via GitHub Actions for automated pipeline execution on push
- **Multi-dataset training** (FD002, FD003, FD004) for generalization

---

*Report generated: 2026-05-15 | Pipeline run time: 0:02:10*
