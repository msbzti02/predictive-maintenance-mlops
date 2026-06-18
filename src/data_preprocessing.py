import pandas as pd
import numpy as np
import os


def load_data(data_path, dataset_id="FD001"):
    cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    print(f"Loading raw text files from {data_path}...")
    train_df = pd.read_csv(os.path.join(data_path, f'train_{dataset_id}.txt'), sep=r'\s+', names=cols)
    test_df  = pd.read_csv(os.path.join(data_path, f'test_{dataset_id}.txt'),  sep=r'\s+', names=cols)
    rul_df   = pd.read_csv(os.path.join(data_path, f'RUL_{dataset_id}.txt'),   sep=r'\s+', names=['RUL'])
    return train_df, test_df, rul_df


def calculate_rul_train(train_df):
    rul = pd.DataFrame(train_df.groupby('engine_id')['cycle'].max()).reset_index()
    rul.columns = ['engine_id', 'max_cycle']
    train_df = train_df.merge(rul, on=['engine_id'], how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop('max_cycle', axis=1, inplace=True)
    return train_df


def calculate_rul_test(test_df, rul_df):
    rul_df['engine_id'] = rul_df.index + 1
    max_cycle = pd.DataFrame(test_df.groupby('engine_id')['cycle'].max()).reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    rul_df = rul_df.merge(max_cycle, on=['engine_id'], how='left')
    rul_df['failure_cycle'] = rul_df['max_cycle'] + rul_df['RUL']
    test_df = test_df.merge(rul_df[['engine_id', 'failure_cycle']], on=['engine_id'], how='left')
    test_df['RUL'] = test_df['failure_cycle'] - test_df['cycle']
    test_df.drop('failure_cycle', axis=1, inplace=True)
    return test_df


def feature_engineering(df):
    print("Applying time-series feature engineering...")
    cols_to_drop = ['s1', 's5', 's10', 's16', 's18', 's19']
    df = df.drop(columns=cols_to_drop)
    sensors = [f's{i}' for i in range(1, 22) if f's{i}' not in cols_to_drop]
    for sensor in sensors:
        df[f'{sensor}_roll_mean_5'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f'{sensor}_roll_std_5'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        )
    return df


def main():
    print("=" * 60)
    print("STAGE 1: DATA INGESTION AND PREPROCESSING")
    print("=" * 60)

    raw_path       = "data/raw"
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)

    train_raw, test_raw, rul_raw = load_data(raw_path, "FD001")
    print(f"Raw Train Shape: {train_raw.shape}")
    print(f"Raw Test Shape:  {test_raw.shape}")

    train_df = calculate_rul_train(train_raw)
    test_df  = calculate_rul_test(test_raw, rul_raw)

    train_df = feature_engineering(train_df)
    test_df  = feature_engineering(test_df)

    train_path = os.path.join(processed_path, "train.csv")
    test_path  = os.path.join(processed_path, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Processed Train saved -> {train_path} | Shape: {train_df.shape}")
    print(f"Processed Test saved  -> {test_path}  | Shape: {test_df.shape}")
    print("Stage 1 completed.\n")


if __name__ == "__main__":
    main()
