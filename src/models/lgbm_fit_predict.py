# src/models/lgbm_fit_predict.py

from typing import Dict, List
import pandas as pd
import lightgbm as lgb

def fit_predict_lgbm_fixed(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: List[str],
    date_col: str,
    target_col: str,
    best_params: Dict
):
    """
    Treina e prediz com LGBM usando hiperparâmetros fixos (best_params).
    """
    train_sorted = train.sort_values(date_col)
    valid_sorted = valid.sort_values(date_col)

    X_tr = train_sorted[features]
    y_tr = train_sorted[target_col]
    X_va = valid_sorted[features]
    y_va = valid_sorted[target_col]

    mdl = lgb.LGBMRegressor(
        random_state=42,
        **best_params
    )

    mdl.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="smape",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False)
        ]
    )

    # feature importance
    df_feature_importance = pd.DataFrame({
        "feature": mdl.feature_name_,
        "importance": mdl.feature_importances_
    })

    preds = mdl.predict(X_va)
    return preds, mdl, df_feature_importance
