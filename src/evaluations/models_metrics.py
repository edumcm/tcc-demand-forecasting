# src/evaluation/models_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricsResult:
    """Container imutável com as métricas calculadas."""
    rmse: float
    mape: float
    wape: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": float(self.rmse),
            "mape": float(self.mape),
            "wape": float(self.wape),
        }


def _validate_inputs(df: pd.DataFrame, y_true: str, y_pred: str) -> None:
    """Valida existência das colunas necessárias."""
    missing = [c for c in (y_true, y_pred) if c not in df.columns]
    if missing:
        raise ValueError(f"Coluna(s) ausente(s) no DataFrame: {missing}")


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE robusto:
    - ignora linhas onde y_true é NaN
    - ignora linhas onde y_true == 0 (evita divisão por zero e valores explosivos)
    """
    mask = ~np.isnan(y_true) & (y_true != 0)
    if not np.any(mask):
        # Se não houver valores válidos para MAPE, devolve NaN
        return float("nan")
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(ape) * 100.0)


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100

    - Usa todos os pontos válidos (assumindo que NaNs já foram tratados antes).
    - Se a soma de |y_true| for 0, devolve NaN.
    """
    # Máscara de segurança contra NaNs (caso dropna=False no cálculo anterior)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return float("nan")

    y_t = y_true[mask]
    y_p = y_pred[mask]

    denom = np.sum(np.abs(y_t))
    if denom == 0:
        return float("nan")

    num = np.sum(np.abs(y_t - y_p))
    return float(num / denom * 100.0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(np.square(diff))))


def calculate_metrics(
    df: pd.DataFrame,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    dropna: bool = True,
) -> MetricsResult:
    """
    Calcula RMSE, MAPE e WAPE para um DataFrame com coluna de predição e target.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas de target e predição.
    y_true : str, default "y_true"
        Nome da coluna do alvo/valor real.
    y_pred : str, default "y_pred"
        Nome da coluna de predição do modelo.
    dropna : bool, default True
        Se True, remove linhas com NaN em y_true ou y_pred antes do cálculo das métricas.
        (Para o MAPE, y_true == 0 também é automaticamente ignorado.)

    Retorno
    -------
    MetricsResult
        Objeto com rmse, mape e wape.
    """
    _validate_inputs(df, y_true, y_pred)

    data = df[[y_true, y_pred]].copy()

    if dropna:
        data = data.dropna(subset=[y_true, y_pred])

    # Se após o drop ficar vazio, devolve métricas NaN
    if data.empty:
        return MetricsResult(
            rmse=float("nan"),
            mape=float("nan"),
            wape=float("nan"),
        )

    y_t = data[y_true].to_numpy(dtype=float)
    y_p = data[y_pred].to_numpy(dtype=float)

    rmse = _rmse(y_t, y_p)
    mape = _safe_mape(y_t, y_p)
    wape = _wape(y_t, y_p)

    return MetricsResult(rmse=rmse, mape=mape, wape=wape)


def compare_models(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    names: Tuple[str, str] = ("model_1", "model_2"),
) -> pd.DataFrame:
    """
    Compara dois modelos calculando RMSE, MAPE e WAPE de cada um e retornando um DataFrame tidy.

    Parâmetros
    ----------
    df1, df2 : pd.DataFrame
        DataFrames com colunas de alvo e predição.
        Observação: não é necessário que tenham o mesmo número de linhas; a comparação é por métrica agregada.
    y_true : str, default "y_true"
        Nome da coluna de alvo/valor real em ambos os DataFrames.
    y_pred : str, default "y_pred"
        Nome da coluna de predição em ambos os DataFrames.
    names : tuple(str, str), default ("model_1", "model_2")
        Rótulos para identificar cada modelo no resultado.

    Retorno
    -------
    pd.DataFrame
        Tabela com colunas: ["model", "rmse", "mape", "wape"].
    """
    m1 = calculate_metrics(df1, y_true=y_true, y_pred=y_pred)
    m2 = calculate_metrics(df2, y_true=y_true, y_pred=y_pred)

    out = pd.DataFrame(
        [
            {"model": names[0], **m1.to_dict()},
            {"model": names[1], **m2.to_dict()},
        ],
        columns=["model", "rmse", "mape", "wape"],
    )
    return out
