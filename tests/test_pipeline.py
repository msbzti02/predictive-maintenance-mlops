import pandas as pd
import os
import yaml

def test_raw_data_exists():
    """Ensure raw NASA data is present before pipeline runs."""
    assert os.path.exists("data/raw/train_FD001.txt"), "Missing training data"
    assert os.path.exists("data/raw/test_FD001.txt"), "Missing testing data"

def test_processed_data_schema():
    """Ensure feature engineering generated the correct schema and no NaNs."""
    if os.path.exists("data/processed/train.csv"):
        df = pd.read_csv("data/processed/train.csv")
        assert "RUL" in df.columns, "Target variable RUL is missing!"
        assert "engine_id" in df.columns, "engine_id is missing!"
        assert df.isnull().sum().sum() == 0, "Data leakage/NaNs detected in processed data!"
        assert df['RUL'].min() >= 0, "RUL cannot be negative!"

def test_config_validity():
    """Ensure the YAML configuration is valid and readable."""
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "mlflow" in config
    assert "data" in config
