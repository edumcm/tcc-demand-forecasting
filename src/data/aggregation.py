# src/data/aggregation.py
"""
Agregação semanal e criação de features básicas (Olist).

Este módulo lê o parquet intermediário gerado no preprocessing,
calcula deltas de tempo por linha, cria a chave semanal, agrega
métricas por produto e por categoria, gera variáveis regionais
no formato "wide" e também uma versão "weighted" (ponderada por vendas).

Principais saídas:
- data/interim/olist_weekly_agg.parquet  (nível categoria x semana, com features)
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
# Importa utils do loader e paths com fallback seguro (mesma ideia do preprocessing.py)
# -----------------------------------------------------------------------------
try:
    # Import relativo (quando rodando via pacote)
    from .loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
    from .loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
except Exception:
    # Import absoluto (quando rodando via notebook/raiz do projeto)
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
# Constantes
# -----------------------------------------------------------------------------
DEFAULT_INPUT_NAME = "olist_merged.parquet"         # saída do preprocessing
DEFAULT_OUTPUT_NAME = "olist_weekly_agg.parquet"    # saída desta etapa

REGIONS = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
REGION_SUFFIX = {
    "Norte": "n",
    "Nordeste": "ne",
    "Sudeste": "se",
    "Sul": "s",
    "Centro-Oeste": "co",
}

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
# 1) Cálculo de deltas de tempo linha a linha (mais interpretável)
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
# 2) Chave semanal
# -----------------------------------------------------------------------------
def add_order_week(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["order_purchase_timestamp"])
    df = df.copy()
    df["order_week"] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time
    return df

# -----------------------------------------------------------------------------
# 3) Variação de preço por janela (nível produto → sobe para categoria)
# -----------------------------------------------------------------------------
def compute_price_vars_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variações de preço no nível (product_id, order_week).

    Saídas:
      - price_var_w1_point  : variação t vs t-1
      - price_var_w1_smooth : média móvel (janela=2) aplicada sobre price_var_w1_point
      - price_var_w4_point  : variação t vs t-4
      - price_var_m4_vs_prev4: variação entre "mês móvel" (4s) atual vs 4s anterior

    Obs.: Mantemos price_mean no retorno (útil para diagnósticos e pesos).
    """
    _require_columns(df, ["product_id", "order_week", "price"])

    grp = (
        df.groupby(["product_id", "order_week"], as_index=False)
          .agg(price_mean=("price", "mean"))
          .sort_values(["product_id", "order_week"])
    )

    # Variações ponto-a-ponto
    grp["price_var_w1_point"] = grp.groupby("product_id")["price_mean"].pct_change(1)   # t vs t-1
    grp["price_var_w4_point"] = grp.groupby("product_id")["price_mean"].pct_change(4)   # t vs t-4

    # Janela 4 semanas (mês móvel) e comparação de blocos (t..t-3) vs (t-4..t-7)
    grp["price_roll4_mean"] = grp.groupby("product_id")["price_mean"].transform(
        lambda s: s.rolling(4, min_periods=4).mean()
    )
    grp["price_var_m4_vs_prev4"] = grp.groupby("product_id")["price_roll4_mean"].pct_change(4)

    # NOVO: suavização da variação semanal (média móvel da variação, janela=2)
    grp["price_var_w1_smooth"] = (
        grp.groupby("product_id")["price_var_w1_point"]
           .transform(lambda s: s.rolling(window=2, min_periods=1).mean())
    )

    return grp

def aggregate_price_vars_to_category(df: pd.DataFrame, price_vars: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variações de preço do nível produto → categoria por semana,
    usando APENAS MÉDIA PONDERADA por "receita" do produto na semana.

    Definição do peso (por produto x semana):
      - Se existir 'order_item_id': qty = contagem de itens; price_mean = preço médio;
        revenue = qty * price_mean  (peso preferencial)
      - Caso contrário:
          • Se existir 'order_id': qty = pedidos únicos; revenue = qty * price_mean
          • Fallback: revenue = soma de 'price' (quando não há qty claro)

    Saída no nível (product_category_name, order_week) com colunas:
      - price_var_w1_point_mean       (PONDERADA)
      - price_var_w1_smooth_mean      (PONDERADA)
      - price_var_w4_point_mean       (PONDERADA)
      - price_var_m4_vs_prev4_mean    (PONDERADA)

    IMPORTANTE: Mantemos o sufixo `_mean` por compatibilidade com o pipeline,
    mas estas colunas agora representam **médias ponderadas** (por receita).
    """
    _require_columns(df, ["product_id", "product_category_name", "order_week", "price"])
    _require_columns(
        price_vars,
        [
            "product_id", "order_week",
            "price_var_w1_point", "price_var_w1_smooth",
            "price_var_w4_point", "price_var_m4_vs_prev4",
            "price_mean"  # veio de compute_price_vars_by_product
        ],
    )

    # --------------------------
    # 1) Construção dos pesos
    # --------------------------
    # Base para pesos no nível produto x semana
    if "order_item_id" in df.columns:
        w = (
            df.groupby(["product_id", "order_week"], as_index=False)
              .agg(
                  qty=("order_item_id", "count"),
                  price_mean_raw=("price", "mean")
              )
        )
        w["revenue"] = w["qty"] * w["price_mean_raw"].astype(float)

    elif "order_id" in df.columns:
        w = (
            df.groupby(["product_id", "order_week"], as_index=False)
              .agg(
                  qty=("order_id", "nunique"),
                  price_mean_raw=("price", "mean")
              )
        )
        w["revenue"] = w["qty"] * w["price_mean_raw"].astype(float)
    else:
        # Fallback: usa soma de price como "proxy" de receita
        w = (
            df.groupby(["product_id", "order_week"], as_index=False)
              .agg(revenue=("price", "sum"))
        )
        # Garante existência das colunas (mesmo que não usadas)
        w["qty"] = np.nan
        w["price_mean_raw"] = np.nan

    # Junta pesos às variações
    pv = price_vars.merge(w[["product_id", "order_week", "revenue"]], on=["product_id", "order_week"], how="left")

    # Junta categoria para poder agregar
    pv = pv.merge(
        df[["product_id", "product_category_name", "order_week"]].drop_duplicates(),
        on=["product_id", "order_week"],
        how="left",
    )

    # Helper para média ponderada robusta
    def _wavg(values: pd.Series, weights: pd.Series) -> float:
        v = values.astype(float)
        w = weights.astype(float)
        # limpa casos inválidos
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return np.nan
        return float(np.average(v[mask], weights=w[mask]))

    # --------------------------
    # 2) Agregação ponderada por categoria x semana
    # --------------------------
    agg = (
        pv.groupby(["product_category_name", "order_week"])
          .apply(lambda g: pd.Series({
              # OBS: apesar do sufixo _mean, são médias PONDERADAS por 'revenue'
              "price_var_w1_point_mean":    _wavg(g["price_var_w1_point"],    g["revenue"]),
              "price_var_w1_smooth_mean":   _wavg(g["price_var_w1_smooth"],   g["revenue"]),
              "price_var_w4_point_mean":    _wavg(g["price_var_w4_point"],    g["revenue"]),
              "price_var_m4_vs_prev4_mean": _wavg(g["price_var_m4_vs_prev4"], g["revenue"]),
          }))
          .reset_index()
          .sort_values(["product_category_name", "order_week"])
          .reset_index(drop=True)
    )

    return agg

# -----------------------------------------------------------------------------
# 4) Agregações por categoria x região x semana -> WIDE + WEIGHTED
# -----------------------------------------------------------------------------
def aggregate_time_metrics_wide_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega (categoria, região, semana):
      - delivery_diff_estimated_mean
      - est_delivery_lead_days_mean
      - approval_time_hours_mean
      - sales (peso: número de itens/pedidos na semana)

    Retorna DF no nível (categoria, semana) com colunas:
      - *_<sufixo_regional>  (wide)
      - *_weighted           (média ponderada pelas vendas regionais)
    """
    _require_columns(
        df,
        [
            "product_category_name",
            "customer_region",
            "order_week",
            "delivery_diff_estimated",
            "estimated_delivery_lead_days",
            "approval_time_hours",
        ],
    )

    # Peso de vendas: preferimos número de itens (order_item_id), se existir;
    # caso contrário, contamos linhas.
    if "order_item_id" in df.columns:
        df["_one"] = 1
        sales_weight = ("_one", "sum")
    else:
        sales_weight = ("order_id", "nunique") if "order_id" in df.columns else ("customer_region", "size")

    per_reg = (
        df.groupby(["product_category_name", "customer_region", "order_week"], as_index=False)
          .agg(
              delivery_diff_estimated_mean=("delivery_diff_estimated", "mean"),
              est_delivery_lead_days_mean=("estimated_delivery_lead_days", "mean"),
              approval_time_hours_mean=("approval_time_hours", "mean"),
              sales=sales_weight,
          )
    )
    # renomear a coluna de vendas (caso venha com tuple)
    if isinstance(per_reg.columns[-1], tuple):
        per_reg = per_reg.rename(columns={per_reg.columns[-1]: "sales"})

    # --- WIDE ---
    wide = _safe_pivot_wide(
        per_reg,
        index_cols=["product_category_name", "order_week"],
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
        per_reg.groupby(["product_category_name", "order_week"])
               .apply(_weighted_row)
               .reset_index()
    )

    out = wide.merge(weighted, on=["product_category_name", "order_week"], how="left")
    return out

# -----------------------------------------------------------------------------
# 5) Target e base final
# -----------------------------------------------------------------------------
def aggregate_sales_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a quantidade semanal vendida por categoria (sales_qty).
    Preferência por contagem de itens (order_item_id) se existir.
    """
    _require_columns(df, ["product_category_name", "order_week"])

    if "order_item_id" in df.columns:
        # cada linha é um item do pedido
        tgt = (
            df.groupby(["product_category_name", "order_week"], as_index=False)
              .agg(sales_qty=("order_item_id", "count"))
        )
    else:
        # fallback: conta pedidos únicos
        col = "order_id" if "order_id" in df.columns else "product_category_name"
        agg_fn = "nunique" if col == "order_id" else "size"
        tgt = (
            df.groupby(["product_category_name", "order_week"], as_index=False)
              .agg(sales_qty=(col, agg_fn))
        )
    return tgt

def build_weekly_aggregation(
    interim_dir: Optional[str | os.PathLike] = None,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    project_dir: Optional[str | os.PathLike] = None,
) -> Path:
    """
    Pipeline completo:
      1) Lê parquet intermediário do preprocessing.
      2) Calcula deltas de tempo e chave semanal.
      3) Calcula variação de preço por janela no nível do produto e agrega para categoria.
      4) Agrega métricas de tempo por categoria x região (WIDE + WEIGHTED).
      5) Agrega target (sales_qty) por categoria x semana.
      6) Mescla tudo no nível (categoria, semana) e salva parquet.
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

    # 2) Chave semanal
    LOGGER.info("Adicionando chave semanal (order_week) ...")
    df = add_order_week(df)

    # 3) Variação de preço (produto -> categoria)
    LOGGER.info("Calculando variação de preço (produto) e agregando para categoria ...")
    price_vars = compute_price_vars_by_product(df)
    price_cat = aggregate_price_vars_to_category(df, price_vars)

    # 4) Métricas de tempo por região (WIDE + WEIGHTED)
    LOGGER.info("Agregando métricas de tempo por categoria x região (wide + weighted) ...")
    time_feats = aggregate_time_metrics_wide_weighted(df)

    # 5) Target de vendas por categoria x semana
    LOGGER.info("Agregando target (sales_qty) por categoria x semana ...")
    target = aggregate_sales_target(df)

    # 6) Mescla final
    LOGGER.info("Mesclando blocos de features no nível categoria x semana ...")
    final = (
        target.merge(price_cat, on=["product_category_name", "order_week"], how="left")
              .merge(time_feats, on=["product_category_name", "order_week"], how="left")
              .sort_values(["product_category_name", "order_week"])
              .reset_index(drop=True)
    )

    out_path = Path(interim).joinpath(output_name).resolve()
    final.to_parquet(out_path, index=False)
    LOGGER.info("Agregação semanal gerada: %s (linhas=%d, colunas=%d)", out_path, len(final), final.shape[1])

    return out_path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gera agregação semanal por categoria com features básicas (Olist).")
    p.add_argument("--interim", dest="interim", default=None, help="Diretório dos parquets intermediários")
    p.add_argument("--input", dest="input", default=DEFAULT_INPUT_NAME, help="Nome do parquet de entrada (preprocessing)")
    p.add_argument("--output", dest="output", default=DEFAULT_OUTPUT_NAME, help="Nome do parquet de saída")
    p.add_argument("--project", dest="project", default=None, help="Diretório raiz do projeto (opcional)")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    build_weekly_aggregation(
        interim_dir=args.interim,
        input_name=args.input,
        output_name=args.output,
        project_dir=args.project,
    )

if __name__ == "__main__":
    main()
