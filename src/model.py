import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_feature_cols(df, target_col):
    drop_cols = [
        "Date",
        "Revenue",
        "COGS",
        "is_train",
        target_col,
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    return feature_cols


def transform_target(y):
    return np.log1p(y.clip(lower=0))


def inverse_transform_target(y_pred):
    return np.clip(np.expm1(y_pred), 0, None)


def raw_mae_eval(y_true, y_pred):
    y_true_raw = inverse_transform_target(y_true)
    y_pred_raw = inverse_transform_target(y_pred)
    mae = np.mean(np.abs(y_true_raw - y_pred_raw))
    return "raw_mae", mae, False


def train_lgbm(train_df, valid_df, target_col, params):
    feature_cols = get_feature_cols(train_df, target_col)

    X_train = train_df[feature_cols]
    y_train = transform_target(train_df[target_col])

    X_valid = valid_df[feature_cols]
    y_valid = transform_target(valid_df[target_col])

    model = lgb.LGBMRegressor(**params, importance_type="gain")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=raw_mae_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model, feature_cols


def predict_lgbm(model, df, feature_cols):
    X = df[feature_cols]

    best_iter = getattr(model, "best_iteration_", None)

    if best_iter is not None and best_iter <= 0:
        best_iter = None

    pred_log = model.predict(X, num_iteration=best_iter)

    pred = inverse_transform_target(pred_log)

    return pred


def train_ridge(train_df, valid_df, target_col, alpha=1.0):
    feature_cols = get_feature_cols(train_df, target_col)

    X_train = train_df[feature_cols]
    y_train = transform_target(train_df[target_col])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(X_train, y_train)

    return model, feature_cols


def predict_ridge(model, df, feature_cols):
    X = df[feature_cols]

    pred_log = model.predict(X)

    pred = inverse_transform_target(pred_log)

    return pred


def train_final_model(train_df, target_col, params, feature_cols=None):
    if feature_cols is None:
        feature_cols = get_feature_cols(train_df, target_col)

    X_train = train_df[feature_cols]
    y_train = transform_target(train_df[target_col])

    model = lgb.LGBMRegressor(**params, importance_type="gain")
    model.fit(X_train, y_train)

    fi_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, feature_cols, fi_df
