# src/data/aggregation.py
"""
Agregação temporal (semanal ou diária) e criação de features básicas (Olist).

Este módulo lê o parquet intermediário gerado no preprocessing,
calcula deltas de tempo por linha, cria a chave de agregação temporal
(semana ou dia), agrega métricas por produto e por categoria, gera
variáveis regionais no formato "wide" e também uma versão "weighted"
(ponderada por vendas), e por fim consolida tudo em um dataset
no nível de tempo (sem categoria).

Principais saídas:
- data/interim/olist_agg.parquet          (nível tempo, com features)
- ou o nome passado via --output
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
DEFAULT_INPUT_NAME = "olist_merged.parquet"      # saída do preprocessing
DEFAULT_OUTPUT_NAME = "olist_weekly_agg.parquet" # saída desta etapa (padrão: semanal)

REGIONS = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
REGION_SUFFIX = {
    "Norte": "n",
    "Nordeste": "ne",
    "Sudeste": "se",
    "Sul": "s",
    "Centro-Oeste": "co",
}

# -----------------------------------------------------------------------------
# Importa utils do loader e paths com fallback seguro
# -----------------------------------------------------------------------------
try:
    from .loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
    from .loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
except Exception:
    try:
        from src.data.loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
        from src.data.loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
    except Exception:
        _PROJECT_DIR = None
        _INTERIM_DIR = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    _h.setFormatter(logging.Formatter(_fmt))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Helpers de caminho 
# -----------------------------------------------------------------------------
def _as_project_dir(explicit: Optional[str | os.PathLike] = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if _PROJECT_DIR:
        return Path(_PROJECT_DIR).resolve()
    # fallback: src/data/aggregation.py -> ... -> raiz do projeto
    return Path(__file__).resolve().parents[2]

def _as_interim_dir(explicit: Optional[str | os.PathLike], project_dir: Path) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if _INTERIM_DIR:
        return Path(_INTERIM_DIR).resolve()
    return project_dir.joinpath("data", "interim").resolve()

# -----------------------------------------------------------------------------
# Helpers gerais
# -----------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {missing}")

def _safe_pivot_wide(
    df: pd.DataFrame,
    index_cols: List[str],
    column: str,
    values: List[str],
    suffix_map: Dict[str, str],
) -> pd.DataFrame:
    """Pivot com múltiplos valores e sufixo curto por região."""
    wide = df.pivot_table(index=index_cols, columns=column, values=values)
    # Flatten de colunas multiindex -> nome_metrica_sufixo
    new_cols = []
    for val_name, reg in wide.columns.to_flat_index():
        suf = suffix_map.get(reg, reg.lower())
        new_cols.append(f"{val_name}_{suf}")
    wide.columns = new_cols
    return wide.reset_index()

# -----------------------------------------------------------------------------
# 1) Cálculo de deltas de tempo linha a linha
# -----------------------------------------------------------------------------
def calc_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém:
      - delivery_diff_estimated (dias) = estimado - entregue
      - estimated_delivery_lead_days (dias) = estimado - compra
      - approval_time_hours (horas) = aprovado - compra
    """
    _require_columns(
        df,
        [
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
            "order_purchase_timestamp",
            "order_approved_at",
        ],
    )

    df = df.copy()
    df["delivery_diff_estimated"] = (
        (df["order_estimated_delivery_date"] - df["order_delivered_customer_date"]).dt.days
    )

    df["estimated_delivery_lead_days"] = (
        (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    )

    df["approval_time_hours"] = (
        (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600.0
    )

    return df

# -----------------------------------------------------------------------------
# 2) Chave temporal (diária ou semanal)
# -----------------------------------------------------------------------------
def add_time_key(df: pd.DataFrame, freq: str = "W") -> tuple[pd.DataFrame, str]:
    """
    Adiciona coluna de agregação temporal a partir de order_purchase_timestamp.

    freq:
      - "W": agrega por semana (start_time da semana ISO) -> coluna "order_week"
      - "D": agrega por dia (data) -> coluna "order_date"

    Retorna (df_novo, nome_coluna_chave)
    """
    _require_columns(df, ["order_purchase_timestamp"])
    freq = freq.upper()
    df = df.copy()

    if freq == "W":
        key_col = "order_week"
        df[key_col] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time
    elif freq == "D":
        key_col = "order_date"
        df[key_col] = df["order_purchase_timestamp"].dt.floor("D")
    else:
        raise ValueError("freq deve ser 'W' (semanal) ou 'D' (diária).")

    return df, key_col

# -----------------------------------------------------------------------------
# 3) Variação de preço por janela (nível produto → categoria → tempo global)
# -----------------------------------------------------------------------------
def compute_price_vars_by_product(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Calcula variações de preço no nível (product_id, time_col).

    Saídas:
      - price_var_w1_point  : variação t vs t-1
      - price_var_w1_smooth : média móvel (janela=2) aplicada sobre price_var_w1_point
      - price_var_w4_point  : variação t vs t-4
      - price_var_m4_vs_prev4: variação entre "mês móvel" (4 unidades) atual vs 4 anteriores

    Obs.: Mantemos price_mean no retorno (útil para diagnósticos e pesos).
    """
    _require_columns(df, ["product_id", time_col, "price"])

    grp = (
        df.groupby(["product_id", time_col], as_index=False)
          .agg(price_mean=("price", "mean"))
          .sort_values(["product_id", time_col])
    )

    # Variações ponto-a-ponto
    grp["price_var_w1_point"] = grp.groupby("product_id")["price_mean"].pct_change(1)
    grp["price_var_w4_point"] = grp.groupby("product_id")["price_mean"].pct_change(4)

    # Janela 4 períodos (podem ser 4 dias ou 4 semanas) e comparação de blocos
    grp["price_roll4_mean"] = grp.groupby("product_id")["price_mean"].transform(
        lambda s: s.rolling(4, min_periods=4).mean()
    )
    grp["price_var_m4_vs_prev4"] = grp.groupby("product_id")["price_roll4_mean"].pct_change(4)

    # Suavização da variação de 1 período (média móvel da variação, janela=2)
    grp["price_var_w1_smooth"] = (
        grp.groupby("product_id")["price_var_w1_point"]
           .transform(lambda s: s.rolling(window=2, min_periods=1).mean())
    )

    return grp

def aggregate_price_vars_to_category(
    df: pd.DataFrame,
    price_vars: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    """
    Agrega variações de preço do nível produto → categoria por período (dia/semana),
    usando MÉDIA PONDERADA por "receita" do produto no período.

    Saída no nível (product_category_name, time_col) com colunas:
      - price_var_w1_point_mean
      - price_var_w1_smooth_mean
      - price_var_w4_point_mean
      - price_var_m4_vs_prev4_mean
    """
    _require_columns(
        df,
        ["product_id", "product_category_name", time_col, "price"],
    )
    _require_columns(
        price_vars,
        [
            "product_id",
            time_col,
            "price_var_w1_point",
            "price_var_w1_smooth",
            "price_var_w4_point",
            "price_var_m4_vs_prev4",
            "price_mean",
        ],
    )

    # pesos no nível produto x período
    if "order_item_id" in df.columns:
        w = (
            df.groupby(["product_id", time_col], as_index=False)
              .agg(
                  qty=("order_item_id", "count"),
                  price_mean_raw=("price", "mean"),
              )
        )
        w["revenue"] = w["qty"] * w["price_mean_raw"].astype(float)
    elif "order_id" in df.columns:
        w = (
            df.groupby(["product_id", time_col], as_index=False)
              .agg(
                  qty=("order_id", "nunique"),
                  price_mean_raw=("price", "mean"),
              )
        )
        w["revenue"] = w["qty"] * w["price_mean_raw"].astype(float)
    else:
        w = (
            df.groupby(["product_id", time_col], as_index=False)
              .agg(revenue=("price", "sum"))
        )
        w["qty"] = np.nan
        w["price_mean_raw"] = np.nan

    pv = price_vars.merge(
        w[["product_id", time_col, "revenue"]],
        on=["product_id", time_col],
        how="left",
    )

    # adiciona categoria
    pv = pv.merge(
        df[["product_id", "product_category_name", time_col]].drop_duplicates(),
        on=["product_id", time_col],
        how="left",
    )

    def _wavg(values: pd.Series, weights: pd.Series) -> float:
        v = values.astype(float)
        w = weights.astype(float)
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return np.nan
        return float(np.average(v[mask], weights=w[mask]))

    agg = (
        pv.groupby(["product_category_name", time_col])
          .apply(lambda g: pd.Series(
              {
                  "price_var_w1_point_mean":    _wavg(g["price_var_w1_point"],    g["revenue"]),
                  "price_var_w1_smooth_mean":   _wavg(g["price_var_w1_smooth"],   g["revenue"]),
                  "price_var_w4_point_mean":    _wavg(g["price_var_w4_point"],    g["revenue"]),
                  "price_var_m4_vs_prev4_mean": _wavg(g["price_var_m4_vs_prev4"], g["revenue"]),
              }
          ))
          .reset_index()
          .sort_values(["product_category_name", time_col])
          .reset_index(drop=True)
    )

    return agg

def aggregate_price_vars_to_time(
    df: pd.DataFrame,
    price_cat: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    """
    Agrega variações de preço do nível categoria → período global (dia/semana),
    usando média ponderada por receita da categoria no período.

    Saída no nível time_col (uma linha por dia/semana).
    """
    _require_columns(df, ["product_category_name", time_col, "price"])

    # pesos por categoria x período
    if "order_item_id" in df.columns:
        w_cat = (
            df.groupby(["product_category_name", time_col], as_index=False)
              .agg(
                  qty=("order_item_id", "count"),
                  price_mean_raw=("price", "mean"),
              )
        )
        w_cat["revenue_cat"] = w_cat["qty"] * w_cat["price_mean_raw"].astype(float)
    elif "order_id" in df.columns:
        w_cat = (
            df.groupby(["product_category_name", time_col], as_index=False)
              .agg(
                  qty=("order_id", "nunique"),
                  price_mean_raw=("price", "mean"),
              )
        )
        w_cat["revenue_cat"] = w_cat["qty"] * w_cat["price_mean_raw"].astype(float)
    else:
        w_cat = (
            df.groupby(["product_category_name", time_col], as_index=False)
              .agg(revenue_cat=("price", "sum"))
        )
        w_cat["qty"] = np.nan
        w_cat["price_mean_raw"] = np.nan

    pv_cat = price_cat.merge(
        w_cat[["product_category_name", time_col, "revenue_cat"]],
        on=["product_category_name", time_col],
        how="left",
    )

    def _wavg(values: pd.Series, weights: pd.Series) -> float:
        v = values.astype(float)
        w = weights.astype(float)
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return np.nan
        return float(np.average(v[mask], weights=w[mask]))

    agg_time = (
        pv_cat.groupby(time_col)
              .apply(lambda g: pd.Series(
                  {
                      "price_var_w1_point_mean":    _wavg(g["price_var_w1_point_mean"],    g["revenue_cat"]),
                      "price_var_w1_smooth_mean":   _wavg(g["price_var_w1_smooth_mean"],   g["revenue_cat"]),
                      "price_var_w4_point_mean":    _wavg(g["price_var_w4_point_mean"],    g["revenue_cat"]),
                      "price_var_m4_vs_prev4_mean": _wavg(g["price_var_m4_vs_prev4_mean"], g["revenue_cat"]),
                  }
              ))
              .reset_index()
              .sort_values(time_col)
              .reset_index(drop=True)
    )

    return agg_time

# -----------------------------------------------------------------------------
# 4) Agregações por região x tempo -> WIDE + WEIGHTED (nível global)
# -----------------------------------------------------------------------------
def aggregate_time_metrics_wide_weighted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Agrega (região, período) para métricas de tempo:
      - delivery_diff_estimated_mean
      - est_delivery_lead_days_mean
      - approval_time_hours_mean
      - sales (peso: número de itens/pedidos no período)

    Retorna DF no nível time_col com colunas:
      - *_<sufixo_regional>  (wide por região)
      - *_weighted           (média ponderada pelas vendas regionais)
    """
    _require_columns(
        df,
        [
            "customer_region",
            time_col,
            "delivery_diff_estimated",
            "estimated_delivery_lead_days",
            "approval_time_hours",
        ],
    )

    # Peso de vendas: preferimos número de itens (order_item_id), se existir;
    # caso contrário, contamos pedidos únicos ou linhas.
    if "order_item_id" in df.columns:
        df_tmp = df.copy()
        df_tmp["_one"] = 1
        sales_weight = ("_one", "sum")
    else:
        sales_weight = ("order_id", "nunique") if "order_id" in df.columns else ("customer_region", "size")
        df_tmp = df

    per_reg = (
        df_tmp.groupby(["customer_region", time_col], as_index=False)
              .agg(
                  delivery_diff_estimated_mean=("delivery_diff_estimated", "mean"),
                  est_delivery_lead_days_mean=("estimated_delivery_lead_days", "mean"),
                  approval_time_hours_mean=("approval_time_hours", "mean"),
                  sales=sales_weight,
              )
    )
    # renomeia a coluna de vendas se ela vier como tupla
    if isinstance(per_reg.columns[-1], tuple):
        per_reg = per_reg.rename(columns={per_reg.columns[-1]: "sales"})

    # --- WIDE (por região, período) ---
    wide = _safe_pivot_wide(
        per_reg,
        index_cols=[time_col],
        column="customer_region",
        values=[
            "delivery_diff_estimated_mean",
            "est_delivery_lead_days_mean",
            "approval_time_hours_mean",
        ],
        suffix_map=REGION_SUFFIX,
    )

    # --- WEIGHTED (ponderada por participação das vendas regionais) ---
    def _weighted_row(group: pd.DataFrame) -> pd.Series:
        tot = group["sales"].sum()
        if tot <= 0 or not np.isfinite(tot):
            return pd.Series(
                {
                    "delivery_diff_estimated_weighted": np.nan,
                    "est_delivery_lead_days_weighted": np.nan,
                    "approval_time_hours_weighted": np.nan,
                }
            )
        w = group["sales"].values.astype(float)
        return pd.Series(
            {
                "delivery_diff_estimated_weighted": np.average(group["delivery_diff_estimated_mean"].values, weights=w),
                "est_delivery_lead_days_weighted": np.average(group["est_delivery_lead_days_mean"].values, weights=w),
                "approval_time_hours_weighted": np.average(group["approval_time_hours_mean"].values, weights=w),
            }
        )

    weighted = (
        per_reg.groupby(time_col)
               .apply(_weighted_row)
               .reset_index()
    )

    out = wide.merge(weighted, on=time_col, how="left")
    return out

# -----------------------------------------------------------------------------
# 5) Target e base final (nível temporal global)
# -----------------------------------------------------------------------------
def aggregate_sales_target(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Agrega a quantidade vendida por período (dia/semana) para todo o portfólio.
    Preferência por contagem de itens (order_item_id) se existir.
    """
    _require_columns(df, [time_col])

    if "order_item_id" in df.columns:
        tgt = (
            df.groupby([time_col], as_index=False)
              .agg(sales_qty=("order_item_id", "count"))
        )
    else:
        col = "order_id" if "order_id" in df.columns else time_col
        agg_fn = "nunique" if col == "order_id" else "size"
        tgt = (
            df.groupby([time_col], as_index=False)
              .agg(sales_qty=(col, agg_fn))
        )
    return tgt

# -----------------------------------------------------------------------------
# 6) Pipeline principal parametrizável (freq = 'W' ou 'D')
# -----------------------------------------------------------------------------
def build_aggregation(
    interim_dir: Optional[str | os.PathLike] = None,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    project_dir: Optional[str | os.PathLike] = None,
    freq: str = "W",
) -> Path:
    """
    Pipeline completo (parametrizável para frequência semanal ou diária):

      1) Lê parquet intermediário do preprocessing.
      2) Calcula deltas de tempo.
      3) Cria chave temporal (semana ou dia).
      4) Calcula variação de preço no nível do produto e agrega:
         - produto → categoria (ponderado)
         - categoria → período global (ponderado).
      5) Agrega métricas de tempo por região (WIDE + WEIGHTED) no nível do período.
      6) Agrega target (sales_qty) por período.
      7) Mescla tudo no nível temporal (sem categoria) e salva parquet.

    freq:
      - "W": agregação semanal (padrão).
      - "D": agregação diária.
    """
    proj_dir = _as_project_dir(project_dir)
    interim = _as_interim_dir(interim_dir, proj_dir)
    Path(interim).mkdir(parents=True, exist_ok=True)

    in_path = Path(interim).joinpath(input_name).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet de entrada não encontrado: {in_path}")

    LOGGER.info("Lendo parquet intermediário: %s", in_path)
    df = pd.read_parquet(in_path)

    # Garantias mínimas
    needed = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
        "order_approved_at",
        "product_id",
        "product_category_name",
        "price",
        "customer_region",
    ]
    _require_columns(df, needed)

    # 1) Deltas de tempo
    LOGGER.info("Calculando deltas de tempo por linha ...")
    df = calc_time_deltas(df)

    # 2) Chave temporal
    LOGGER.info("Adicionando chave temporal (%s) ...", freq)
    df, time_col = add_time_key(df, freq=freq)

    # 3) Variação de preço (produto -> categoria -> período global)
    LOGGER.info("Calculando variação de preço no nível produto ...")
    price_vars = compute_price_vars_by_product(df, time_col=time_col)

    LOGGER.info("Agregando variação de preço para categoria (ponderado) ...")
    price_cat = aggregate_price_vars_to_category(df, price_vars, time_col=time_col)

    LOGGER.info("Agregando variação de preço de categoria para nível temporal global (ponderado) ...")
    price_time = aggregate_price_vars_to_time(df, price_cat, time_col=time_col)

    # 4) Métricas de tempo por região (WIDE + WEIGHTED) no nível temporal
    LOGGER.info("Agregando métricas de tempo por região x período (wide + weighted) ...")
    time_feats = aggregate_time_metrics_wide_weighted(df, time_col=time_col)

    # 5) Target de vendas por período
    LOGGER.info("Agregando target (sales_qty) por período ...")
    target = aggregate_sales_target(df, time_col=time_col)

    # 6) Mescla final (nível time_col)
    LOGGER.info("Mesclando blocos de features no nível temporal (freq=%s) ...", freq)
    final = (
        target
        .merge(price_time, on=time_col, how="left")
        .merge(time_feats, on=time_col, how="left")
        .sort_values([time_col])
        .reset_index(drop=True)
    )

    out_path = Path(interim).joinpath(output_name).resolve()
    final.to_parquet(out_path, index=False)
    LOGGER.info(
        "Agregação temporal gerada (freq=%s): %s (linhas=%d, colunas=%d)",
        freq,
        out_path,
        len(final),
        final.shape[1],
    )

    return out_path

# -----------------------------------------------------------------------------
# Wrappers para compatibilidade
# -----------------------------------------------------------------------------
def build_weekly_aggregation(
    interim_dir: Optional[str | os.PathLike] = None,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    project_dir: Optional[str | os.PathLike] = None,
) -> Path:
    """Mantido por compatibilidade: agregação semanal."""
    return build_aggregation(
        interim_dir=interim_dir,
        input_name=input_name,
        output_name=output_name,
        project_dir=project_dir,
        freq="W",
    )

def build_daily_aggregation(
    interim_dir: Optional[str | os.PathLike] = None,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = "olist_daily_agg.parquet",
    project_dir: Optional[str | os.PathLike] = None,
) -> Path:
    """Atalho para agregação diária."""
    return build_aggregation(
        interim_dir=interim_dir,
        input_name=input_name,
        output_name=output_name,
        project_dir=project_dir,
        freq="D",
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gera agregação temporal (semanal ou diária) com features básicas (Olist)."
    )
    p.add_argument("--interim", dest="interim", default=None, help="Diretório dos parquets intermediários")
    p.add_argument("--input", dest="input", default=DEFAULT_INPUT_NAME, help="Nome do parquet de entrada (preprocessing)")
    p.add_argument("--output", dest="output", default=DEFAULT_OUTPUT_NAME, help="Nome do parquet de saída")
    p.add_argument(
        "--freq",
        dest="freq",
        default="W",
        choices=["W", "D", "w", "d"],
        help="Frequência de agregação: 'W' para semanal (padrão), 'D' para diária.",
    )
    p.add_argument("--project", dest="project", default=None, help="Diretório raiz do projeto (opcional)")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    build_aggregation(
        interim_dir=args.interim,
        input_name=args.input,
        output_name=args.output,
        project_dir=args.project,
        freq=str(args.freq).upper(),
    )

if __name__ == "__main__":
    main()
