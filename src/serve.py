from fastapi import FastAPI
import uvicorn
import mlflow
import pandas as pd
import yaml
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life of NASA Turbofan Engines",
)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
model_name  = "PredictiveMaintenanceModel"
model_stage = "Production"
model       = None


@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Loading model '{model_name}' stage='{model_stage}' from MLflow Registry...")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running. Visit /docs for Swagger UI."}


@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Production model not found in MLflow Registry. Run the pipeline first."}
    df = pd.DataFrame([data["features"]])
    prediction   = model.predict(df)
    predicted_rul = float(prediction[0])
    return {
        "predicted_RUL_cycles": round(predicted_rul, 2),
        "maintenance_recommendation": (
            "URGENT - Schedule Maintenance" if predicted_rul < 30 else "Healthy"
        ),
    }


if __name__ == "__main__":
    print("Starting API server...")
    uvicorn.run(app, host=config["serve"]["host"], port=config["serve"]["port"])
