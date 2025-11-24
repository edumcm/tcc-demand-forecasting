"""Funções de pré-processamento para o projeto de previsão de demanda (Olist).

Este módulo carrega os datasets brutos (via `src.data.loader.load_dataset`),
realiza limpeza, merges e conversões de tipo, adiciona a região do cliente e
salva um arquivo parquet intermediário em `data/interim/` para uso nas próximas etapas.
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd

# -----------------------------------------------------------------------------
# Importa funções do loader com fallback seguro
# -----------------------------------------------------------------------------
try:
    from .loader import load_dataset  # type: ignore
    try:
        from .loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
        from .loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
    except Exception:
        _PROJECT_DIR = None
        _INTERIM_DIR = None
except Exception:
    from src.data.loader import load_dataset  # type: ignore
    try:
        from src.data.loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
        from src.data.loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
    except Exception:
        _PROJECT_DIR = None
        _INTERIM_DIR = None

# -----------------------------------------------------------------------------
# Configuração de logging
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    _handler.setFormatter(logging.Formatter(_fmt))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Constantes e configurações padrão
# -----------------------------------------------------------------------------
DEFAULT_MERGED_NAME = "olist_merged.parquet"

ORDERS_COLS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

PRODUCTS_COLS = ["product_id", "product_category_name"]
CUSTOMERS_COLS = ["customer_id", "customer_unique_id", "customer_state"]
DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
]
ID_COLS = ["order_id", "customer_id", "product_id"]

# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------
def _as_project_dir(explicit: Optional[str | os.PathLike] = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if _PROJECT_DIR:
        return Path(_PROJECT_DIR).resolve()
    return Path(__file__).resolve().parents[2]

def _as_interim_dir(explicit: Optional[str | os.PathLike], project_dir: Path) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if _INTERIM_DIR:
        return Path(_INTERIM_DIR).resolve()
    return project_dir.joinpath("data", "interim").resolve()

def _first_key_containing(d: Dict[str, pd.DataFrame], substring: str) -> Optional[str]:
    substring = substring.lower()
    for k in d.keys():
        if substring in k.lower():
            return k
    return None

def _safe_select(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        LOGGER.warning("Colunas ausentes durante a seleção: %s", missing)
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()

def _coerce_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df

def _cast_ids_to_str(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def _add_customer_region(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona uma coluna 'customer_region' com base na sigla do estado do cliente."""
    regioes = {
        'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
        'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
        'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
        'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
        'Sul': ['PR', 'RS', 'SC']
    }

    mapa_estado_regiao = {estado: regiao for regiao, estados in regioes.items() for estado in estados}

    if 'customer_state' in df.columns:
        df['customer_region'] = df['customer_state'].map(mapa_estado_regiao)
    else:
        LOGGER.warning("Coluna 'customer_state' não encontrada no dataframe.")

    return df

def _filter_top7_sp(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas os top 7 produtos em SP"""
    df = df[df['customer_state'] == 'SP'].copy()

    top7 = df.groupby('product_category_name')\
        .agg({'product_id': 'count'})\
            .sort_values('product_id', ascending=False).head(7)\
                .index.tolist()
    
    df_top7_sp = df[df['product_category_name'].isin(top7)]

    return df_top7_sp

# -----------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------
def build_olist_merged(
    cfg_path: str | os.PathLike,
    interim_dir: Optional[str | os.PathLike] = None,
    output_name: str = DEFAULT_MERGED_NAME,
    project_dir: Optional[str | os.PathLike] = None,
    filter_apply: bool = True
) -> Path:
    
    proj_dir = _as_project_dir(project_dir)
    interim = _as_interim_dir(interim_dir, proj_dir)
    Path(interim).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Carregando datasets com cfg: %s", cfg_path)
    dfs: Dict[str, pd.DataFrame] = load_dataset(str(cfg_path), dataset="olist", stage="raw")

    orders_key = _first_key_containing(dfs, "orders")
    items_key = _first_key_containing(dfs, "order_items")
    products_key = _first_key_containing(dfs, "products")
    cust_key = _first_key_containing(dfs, "customers")

    required = {"orders": orders_key, "order_items": items_key, "products": products_key, "customers": cust_key}
    missing = [name for name, key in required.items() if key is None]
    if missing:
        raise KeyError(f"Não foi possível encontrar os datasets necessários: {missing}")

    orders = dfs[orders_key].copy()
    items = dfs[items_key].copy()
    products = dfs[products_key].copy()
    customers = dfs[cust_key].copy()

    orders = _safe_select(orders, ORDERS_COLS)
    products = _safe_select(products, PRODUCTS_COLS)
    customers = _safe_select(customers, CUSTOMERS_COLS)

    for _df in (orders, items, products, customers):
        _cast_ids_to_str(_df, ID_COLS)

    LOGGER.info("Mesclando orders x items ...")
    df = pd.merge(orders, items, on="order_id", how="inner")

    LOGGER.info("Mesclando com products ...")
    df = pd.merge(df, products, on="product_id", how="left")

    LOGGER.info("Mesclando com customers ...")
    df = pd.merge(df, customers, on="customer_id", how="left")

    # Adiciona a região do cliente
    df = _add_customer_region(df)

    LOGGER.info("Convertendo colunas de data e filtrando pedidos entregues...")
    _coerce_datetime(df, DATE_COLS)

    conds = [
        (df.get("order_status").eq("delivered")) if "order_status" in df else True,
        (~df.get("product_category_name").isna()) if "product_category_name" in df else True,
        (~df.get("order_approved_at").isna()) if "order_approved_at" in df else True,
        (~df.get("order_delivered_customer_date").isna()) if "order_delivered_customer_date" in df else True,
        (~df.get("order_delivered_carrier_date").isna()) if "order_delivered_carrier_date" in df else True,
    ]
    mask = pd.Series(True, index=df.index)
    for c in conds:
        if isinstance(c, pd.Series):
            mask &= c
    df = df.loc[mask].copy()

    subset = [c for c in ("order_id", "order_item_id") if c in df.columns]
    if subset:
        antes = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        depois = len(df)
        if depois < antes:
            LOGGER.info("Removidas %d duplicatas", antes - depois)

    # filtrando SP e top 7 produtos
    if filter_apply:
        LOGGER.info("Filtrando apenas as top 7 categorias vendidas em SP...")
        df = _filter_top7_sp(df).copy()

    out_path = Path(interim).joinpath(output_name).resolve()
    df.to_parquet(out_path, index=False)
    LOGGER.info("Arquivo parquet gerado: %s (linhas=%d, colunas=%d)", out_path, len(df), df.shape[1])

    return out_path

# -----------------------------------------------------------------------------
# CLI para execução via terminal ou notebook
# -----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gera dataset intermediário mesclado da Olist.")
    p.add_argument("--cfg", dest="cfg", default=str(Path(_as_project_dir()) / "configs" / "data.yaml"), help="Caminho do data.yaml")
    p.add_argument("--interim", dest="interim", default=None, help="Diretório de saída do parquet")
    p.add_argument("--name", dest="name", default=DEFAULT_MERGED_NAME, help="Nome do arquivo parquet gerado")
    p.add_argument("--project", dest="project", default=None, help="Diretório raiz do projeto (opcional)")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    build_olist_merged(
        cfg_path=args.cfg,
        interim_dir=args.interim,
        output_name=args.name,
        project_dir=args.project,
    )

if __name__ == "__main__":
    main()