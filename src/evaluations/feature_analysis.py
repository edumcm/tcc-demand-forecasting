# src/evaluation/feature_analysis.py
# -*- coding: utf-8 -*-

"""
Análise de features (contínuas) em relação ao target:
- Correlação de Pearson
- Mutual Information (MI)
- Ranking combinado (|r| normalizado + MI normalizada)
- Gráficos (heatmap 1D de Pearson e barras de MI)

Saídas (padrão):
- reports/tables/feature_corr_pearson.csv
- reports/tables/feature_mutual_info.csv
- reports/tables/feature_rank.csv
- reports/figures/feature_corr_heatmap.png
- reports/figures/feature_importance_mi.png
- reports/figures/feature_rank_combo.png
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Importa paths do loader com fallback robusto (mesma filosofia do build.py)
# ---------------------------------------------------------------------------
try:
    # execução como pacote (src/)
    from ..data.loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
except Exception:  # pragma: no cover
    try:
        # execução a partir da raiz do repo / notebooks
        from src.data.loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
    except Exception:  # pragma: no cover
        _PROJECT_DIR = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    _h.setFormatter(logging.Formatter(_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False  # evita duplicação em notebooks

# ---------------------------------------------------------------------------
# Helpers de caminho e validação (alinhados ao build.py)
# ---------------------------------------------------------------------------
def _as_project_dir(explicit: Optional[str | os.PathLike] = None) -> Path:
    """Resolve o diretório do projeto, com os mesmos fallbacks do build.py."""
    if explicit:
        return Path(explicit).resolve()
    if _PROJECT_DIR:
        return Path(_PROJECT_DIR).resolve()
    # fallback: src/evaluation/feature_analysis.py -> ... -> raiz do projeto
    return Path(__file__).resolve().parents[2]

def _as_reports_dir(explicit: Optional[str | os.PathLike], project_dir: Path) -> Path:
    """Resolve o diretório base de reports (default: <project_dir>/reports)."""
    if explicit:
        return Path(explicit).resolve()
    return project_dir.joinpath("reports").resolve()

def _ensure_dirs(paths: Iterable[Path]) -> None:
    """Garante que diretórios existam."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Valida a existência de colunas obrigatórias no DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {missing}")

# ---------------------------------------------------------------------------
# Funções utilitárias de seleção / cálculo
# ---------------------------------------------------------------------------
def _is_binary_series(s: pd.Series, allow_nan: bool = True) -> bool:
    if allow_nan:
        vals = pd.unique(s.dropna())
    else:
        vals = pd.unique(s)
    if len(vals) == 0:
        return False
    return set(vals).issubset({0, 1})

def select_continuous_features(
    df: pd.DataFrame,
    target_col: str,
    min_nonnull_frac: float = 0.6,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cont_candidates: List[str] = []
    n = len(df)

    for c in numeric_cols:
        if c == target_col or c in exclude_cols:
            continue
        nonnull_frac = df[c].notna().mean() if n > 0 else 0.0
        if nonnull_frac < min_nonnull_frac:
            continue
        if _is_binary_series(df[c]):
            continue
        cont_candidates.append(c)

    return cont_candidates

def _pairwise_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

def compute_pearson_corr(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []
    y = df[target_col]
    for col in feature_cols:
        r, p = _pairwise_pearson(df[col], y)
        n_used = int((df[col].notna() & y.notna()).sum())
        rows.append({"feature": col, "pearson_r": r, "pearson_pvalue": p, "n_used": n_used})

    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last").reset_index(drop=True)
    return out

def compute_mutual_information(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    standardize: bool = True,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    y_all = df[target_col]

    for col in feature_cols:
        mask = df[col].notna() & y_all.notna()
        x = df.loc[mask, col].to_numpy().reshape(-1, 1)
        y = y_all.loc[mask].to_numpy()

        if len(y) < 5:
            rows.append({"feature": col, "mi": np.nan, "n_used": int(len(y))})
            continue

        if standardize:
            xs = StandardScaler().fit_transform(x)
            ys = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        else:
            xs, ys = x, y

        mi_val = mutual_info_regression(
            xs,
            ys,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )[0]
        rows.append({"feature": col, "mi": float(mi_val), "n_used": int(len(ys))})

    out = pd.DataFrame(rows).sort_values("mi", ascending=False, na_position="last").reset_index(drop=True)
    return out

# ---------------------------------------------------------------------------
# Helpers de salvamento e plots
# ---------------------------------------------------------------------------
def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    LOGGER.info("Arquivo salvo: %s", path)

def _plot_corr_heatmap(corr_df: pd.DataFrame, fig_path: Path, title: str) -> None:
    df_plot = corr_df.copy()
    df_plot["abs_r"] = df_plot["pearson_r"].abs()
    df_plot = df_plot.sort_values("abs_r", ascending=True)

    plt.figure(figsize=(8, max(4, 0.3 * len(df_plot))))
    plt.barh(df_plot["feature"], df_plot["pearson_r"])
    plt.axvline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    LOGGER.info("Figura salva: %s", fig_path)

def _plot_mi_bar(mi_df: pd.DataFrame, fig_path: Path, title: str) -> None:
    df_plot = mi_df.copy().dropna(subset=["mi"])
    if df_plot.empty:
        LOGGER.warning("Sem valores de MI para plotar.")
        return

    mi_max = df_plot["mi"].max()
    df_plot["mi_norm"] = (df_plot["mi"] / mi_max) if mi_max > 0 else 0.0
    df_plot = df_plot.sort_values("mi_norm", ascending=True)

    plt.figure(figsize=(8, max(4, 0.3 * len(df_plot))))
    plt.barh(df_plot["feature"], df_plot["mi_norm"])
    plt.title(title + " (normalizada)")
    plt.xlabel("MI normalizada (0-1)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    LOGGER.info("Figura salva: %s", fig_path)

def _plot_combo_rank(
    corr_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    fig_path: Path,
    title: str,
) -> pd.DataFrame:
    df = pd.merge(corr_df[["feature", "pearson_r"]], mi_df[["feature", "mi"]], on="feature", how="outer")

    r_abs = df["pearson_r"].abs()
    r_max = np.nanmax(r_abs.values)
    df["r_norm_abs"] = np.where((~np.isnan(r_abs)) & (r_max > 0), r_abs / r_max, np.nan)

    mi_vals = df["mi"]
    mi_max = np.nanmax(mi_vals.values)
    df["mi_norm"] = np.where((~np.isnan(mi_vals)) & (mi_max > 0), mi_vals / mi_max, np.nan)

    df["score"] = np.nanmean(df[["r_norm_abs", "mi_norm"]].values, axis=1)

    df_plot = df.sort_values("score", ascending=True)

    plt.figure(figsize=(8, max(4, 0.3 * len(df_plot))))
    plt.barh(df_plot["feature"], df_plot["score"])
    plt.title(title)
    plt.xlabel("Score combinado (0-1)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    LOGGER.info("Figura salva: %s", fig_path)

    return df.sort_values("score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Função principal (padrão de parâmetros alinhado ao build_features)
# ---------------------------------------------------------------------------
def analyze_continuous_features(
    df: pd.DataFrame,
    target_col: str,
    reports_dir: Optional[str | os.PathLike] = None, 
    project_dir: Optional[str | os.PathLike] = None,
    min_nonnull_frac: float = 0.6,
    n_neighbors_mi: int = 3,
    random_state: int = 42,
    top_n_plots: Optional[int] = None,
    filename_prefix: str = "feature_",  # permite customizar nomes de saída, se desejar
) -> dict:
    """
    Pipeline completo para analisar apenas features contínuas e salvar saídas
    sob <reports_dir>/tables e <reports_dir>/figures.

    Parâmetros (padrão de salvamento alinhado ao build.py):
      - reports_dir: diretório-base de reports (default: <project_dir>/reports)
      - project_dir: diretório raiz do projeto (fallback automático)

    Outras configs:
      - min_nonnull_frac: fração mínima de não-nulos para considerar a feature
      - n_neighbors_mi: k da MI
      - random_state: semente
      - top_n_plots: se definido, limita gráficos às Top-N pelo score combinado
      - filename_prefix: prefixo para os nomes dos arquivos gerados

    Retorna:
      dict com DataFrames e lista de features consideradas.
    """
    # Resolve caminhos ao estilo do build.py
    proj_dir = _as_project_dir(project_dir)
    rep_dir = _as_reports_dir(reports_dir, proj_dir)

    tables_dir = rep_dir.joinpath("tables").resolve()
    figures_dir = rep_dir.joinpath("figures").resolve()
    _ensure_dirs([tables_dir, figures_dir])

    # Garantias mínimas de colunas
    _require_columns(df, [target_col])

    # Seleção das features contínuas
    feature_cols = select_continuous_features(
        df=df,
        target_col=target_col,
        min_nonnull_frac=min_nonnull_frac,
        exclude_cols=[],
    )

    if not feature_cols:
        LOGGER.warning("Nenhuma feature contínua elegível encontrada. Verifique filtros e dados.")
        empty_corr = pd.DataFrame(columns=["feature", "pearson_r", "pearson_pvalue", "n_used"])
        empty_mi = pd.DataFrame(columns=["feature", "mi", "n_used"])
        empty_rank = pd.DataFrame(columns=["feature", "pearson_r", "mi", "r_norm_abs", "mi_norm", "score"])
        return {
            "features_consideradas": [],
            "pearson_df": empty_corr,
            "mi_df": empty_mi,
            "rank_df": empty_rank,
        }

    LOGGER.info("Features contínuas consideradas (%d): %s", len(feature_cols), feature_cols)

    # 1) Pearson
    pearson_df = compute_pearson_corr(df=df, target_col=target_col, feature_cols=feature_cols)
    _save_csv(pearson_df, tables_dir.joinpath(f"{filename_prefix}corr_pearson.csv"))

    # 2) MI
    mi_df = compute_mutual_information(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        standardize=True,
        n_neighbors=n_neighbors_mi,
        random_state=random_state,
    )
    _save_csv(mi_df, tables_dir.joinpath(f"{filename_prefix}mutual_info.csv"))

    # 3) Ranking combinado + figura
    rank_df = _plot_combo_rank(
        corr_df=pearson_df,
        mi_df=mi_df,
        fig_path=figures_dir.joinpath(f"{filename_prefix}rank_combo.png"),
        title="Ranking combinado (|r| normalizado + MI normalizada)",
    )

    rank_df["Classificação Pearson"] = rank_df["pearson_r"].apply(lambda x: "Baixo" if abs(x) < 0.3 else ("Moderado" if abs(x) < 0.5 else "Forte"))
    rank_df["Classificação Mi"] = rank_df["mi"].apply(lambda x: "Baixo" if x < 0.3 else ("Moderado" if x < 0.6 else "Forte"))

    _save_csv(rank_df, tables_dir.joinpath(f"{filename_prefix}rank.csv"))

    # 4) Gráficos individuais (Pearson / MI)
    if top_n_plots is not None and top_n_plots > 0 and not rank_df.empty:
        top_feats = rank_df.head(top_n_plots)["feature"].tolist()
        pearson_plot_df = pearson_df[pearson_df["feature"].isin(top_feats)].copy()
        mi_plot_df = mi_df[mi_df["feature"].isin(top_feats)].copy()
    else:
        pearson_plot_df = pearson_df.copy()
        mi_plot_df = mi_df.copy()

    _plot_corr_heatmap(
        corr_df=pearson_plot_df,
        fig_path=figures_dir.joinpath(f"{filename_prefix}corr_heatmap.png"),
        title="Correlação de Pearson (contínuas)",
    )
    _plot_mi_bar(
        mi_df=mi_plot_df,
        fig_path=figures_dir.joinpath(f"{filename_prefix}importance_mi.png"),
        title="Mutual Information (contínuas)",
    )

    return {
        "features_consideradas": feature_cols,
        "pearson_df": pearson_df,
        "mi_df": mi_df,
        "rank_df": rank_df,
    }
