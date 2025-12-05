# src/models/lgbm_tunning.py

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from lightgbm import LGBMRegressor

# ---------- HPO estático ----------
def run_lgbm_hpo_static(
    df: pd.DataFrame,
    features: List[str],
    date_col: str,
    target_col: str,
    first_train_end: pd.Timestamp,
    n_splits: int = 3,
    n_iter: int = 30,
) -> Dict:
    """
    Faz HPO estático (uma vez só) usando todo o período até first_train_end.
    Retorna best_params para serem reutilizados nos backtests por categoria.
    """
    # Usa apenas dados até o fim do treino estático
    df_train = df[df[date_col] <= first_train_end].sort_values(date_col).copy()

    X = df_train[features]
    y = df_train[target_col]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Espaço de busca reduzido para não explodir o tempo
    param = {
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 8, 12],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_samples": [10, 20, 30],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.0, 0.1],
    }

    base = LGBMRegressor(
        random_state=42,
        n_jobs=-1
    )

    searcher = RandomizedSearchCV(
        estimator=base,
        param_distributions=param,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    searcher.fit(X, y)

    best_params = searcher.best_params_
    print(">>> Best params (HPO estático):")
    print(best_params)

    return best_params