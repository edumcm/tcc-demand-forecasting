# src/models/sarima_fit_predict.py
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_predict_sarima_fixed(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    date_col: str,
    target_col: str,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 7),
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
):
    """
    Treina e prediz com SARIMA (via SARIMAX) usando hiperparâmetros fixos.

    Faz forecast dos próximos N pontos, onde N = len(valid).
    Pressupõe que o período de validação vem logo após o treino.
    """
    # Ordena e define índice como data
    train_sorted = train.sort_values(date_col).set_index(date_col)
    valid_sorted = valid.sort_values(date_col).set_index(date_col)

    y_train = train_sorted[target_col]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        model_fit = model.fit(disp=False)

    # Número de passos à frente = tamanho da janela de validação
    steps = len(valid_sorted)

    # Forecast "n steps ahead" a partir do fim do treino
    preds = model_fit.forecast(steps=steps)

    # Garante que o índice das previsões case com o índice da validação
    preds.index = valid_sorted.index

    return preds.values, model_fit



