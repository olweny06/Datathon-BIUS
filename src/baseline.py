import numpy as np
import pandas as pd


def fit_seasonal_trend_baseline(train_df, target_col):
    df = train_df.copy()

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    annual = df.groupby("year")[target_col].sum()

    yoy = annual.pct_change().dropna()

    if len(yoy) > 0:
        growth = (1 + yoy).prod() ** (1 / len(yoy))
    else:
        growth = 1.0

    annual_mean = df.groupby("year")[target_col].transform("mean")
    df[f"{target_col}_norm"] = df[target_col] / annual_mean.replace(0, np.nan)

    seasonal = df.groupby(["month", "day"])[f"{target_col}_norm"].mean().reset_index()

    base_year = df["year"].max()
    base_level = annual.loc[base_year] / 365

    model = {
        "target_col": target_col,
        "growth": growth,
        "seasonal": seasonal,
        "base_year": base_year,
        "base_level": base_level,
    }

    return model


def predict_seasonal_trend_baseline(model, df):
    target_col = model["target_col"]

    pred_df = df.copy()
    pred_df["year"] = pred_df["Date"].dt.year
    pred_df["month"] = pred_df["Date"].dt.month
    pred_df["day"] = pred_df["Date"].dt.day
    pred_df["years_ahead"] = pred_df["year"] - model["base_year"]

    norm_col = f"{target_col}_norm"

    pred_df = pred_df.merge(model["seasonal"], on=["month", "day"], how="left")

    pred_df[norm_col] = pred_df[norm_col].fillna(1.0)

    pred = (
        model["base_level"]
        * (model["growth"] ** pred_df["years_ahead"])
        * pred_df[norm_col]
    )

    return pred.clip(lower=0).values
