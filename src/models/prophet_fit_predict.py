# src/models/prophet_fit_predict.py

from typing import List, Dict, Optional, Any
import pandas as pd
from prophet import Prophet


def fit_predict_prophet_fixed(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    date_col: str,
    target_col: str,
    features: Optional[List[str]],
    params: Optional[Dict[str, Any]] = None,
):
    """
    Treina e prediz com Prophet usando hiperparâmetros fixos (params).

    - date_col: coluna de datas
    - target_col: coluna target (ex.: 'sales_qty')
    - features: lista de colunas usadas como regressoras externas (podem ser [] ou None)
    - params: dicionário com hiperparâmetros do Prophet
      (ex.: {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05, ...})
    """
    # Garante ordem temporal
    train_sorted = train.sort_values(date_col).copy()
    valid_sorted = valid.sort_values(date_col).copy()

    if features is None:
        features = []

    # Monta dataframes no formato esperado pelo Prophet (ds, y, + regressors)
    df_tr = train_sorted[[date_col, target_col] + features].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    df_va = valid_sorted[[date_col] + features].rename(
        columns={date_col: "ds"}
    )

    # Instancia o modelo com os hiperparâmetros informados
    if params is None:
        mdl = Prophet(weekly_seasonality=True,
                            yearly_seasonality=True,
                            daily_seasonality=False)
    else:
        mdl = Prophet(**params)


    # Adiciona regressoras externas, se houver
    for reg in features:
        mdl.add_regressor(reg)

    # Treina
    mdl.fit(df_tr)

    # Faz previsão para o período de validação
    forecast = mdl.predict(df_va)

    # Prophet devolve várias colunas; usamos yhat como predição
    preds = forecast["yhat"].values

    return preds, mdl

