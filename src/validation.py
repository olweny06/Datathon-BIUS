import numpy as np
import pandas as pd


def build_forecasting_frame(sales):
    """
    Tạo dataframe gồm:
    - train: lấy Revenue/COGS thật từ sales.csv
    - test: lấy Date từ sample_submission.csv, target để NaN
    - is_train: đánh dấu train/test

    Output:
        df gồm các cột Date, Revenue, COGS, is_train
    """
    dataset_path = "D:\\DATATHON\\Datathon-BIUS\\dataset"
    sample_sub = pd.read_csv(
        f"{dataset_path}\\sample_submission.csv", parse_dates=["Date"]
    )
    train = sales.copy()
    test = sample_sub[["Date"]].copy()
    test["Revenue"] = np.nan  # Không để 0.0 để tránh hiểu lầm revenue là 0
    test["COGS"] = np.nan
    return train, test


def time_train_valid_split(train_df, valid_ratio=0.2):
    train_df = train_df.sort_values("Date").reset_index(drop=True)

    split_idx = int(len(train_df) * (1 - valid_ratio))

    train_part = train_df.iloc[:split_idx].copy()
    valid_part = train_df.iloc[split_idx:].copy()

    return train_part, valid_part
