# src/data/imputation.py
"""
Imputação de valores nulos nas features agregadas (Olist).

1) Variações de preço e de vendas (ex.: price_var_*, qty_var_*, sales_*):
   -> imputar 0 (interpretação: "sem variação conhecida").
2) Lags e rollings: manter NaN (janelas iniciais são naturalmente incompletas).
3) Métricas regionais (ex.: *_n, *_ne, *_se, *_s, *_co): manter NaN porque
   NÃO entram no modelo (diagnóstico/interpretação).
4) Criar 'have_nulls' após as imputações, sinalizando se restou NaN em alguma
   feature de ENTRADA do modelo (exclui automaticamente colunas regionais e
   outras que podem ser listadas em EXCLUDE_FROM_HAVE_NULLS).

Compatível com bases temporais agregadas:
- Semanal (coluna 'order_week')
- Diária (coluna 'order_date')
"""

from __future__ import annotations
import argparse
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Importa paths do loader com fallback (mesmo padrão dos outros módulos)
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
# Constantes / Config
# -----------------------------------------------------------------------------
# (padrões pensados para fluxo semanal; para diário basta passar outros nomes)
DEFAULT_INPUT_NAME = "olist_weekly_agg_withlags.parquet"
DEFAULT_OUTPUT_NAME = "olist_weekly_agg_withlags_imputed.parquet"

# Prefixos que identificam "variações" (preço e vendas) para imputar 0
VARIATION_PREFIXES = [
    "price_var",        # ex.: price_var_m4_vs_prev4_mean, price_var_lag1 ...
    "qty_var",          # ex.: qty_var_m4_vs_prev4_mean ...
    "sales",            # ex.: sales_qty, sales_*  (pode ser tratado como "sem venda" = 0)
]

# Padrões para identificar colunas regionais que NÃO entram no modelo
# (excluídas do cálculo de have_nulls)
REGIONAL_SUFFIXES = ("_n", "_ne", "_se", "_s", "_co")
REGIONAL_REGEX = re.compile(rf"({'|'.join(s.strip('_') for s in REGIONAL_SUFFIXES)})$")

# Colunas que certamente não devem entrar no cálculo de have_nulls
EXCLUDE_FROM_HAVE_NULLS = {
    # chaves temporais (se estiverem presentes)
    "order_week",
    "order_date",
    "year_week",
    # alvo / métricas agregadas principais
    "sales_qty",
    "revenue",
    # (product_category_name foi removida pois a base final não é mais por categoria)
}

# -----------------------------------------------------------------------------
# Helpers de caminho / validação
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

def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {missing}")

def _is_regional(col: str) -> bool:
    """Retorna True se a coluna aparenta ser regional (sufixos _n, _ne, _se, _s, _co)."""
    return bool(REGIONAL_REGEX.search(col))

def _variation_columns(df: pd.DataFrame) -> list[str]:
    """Seleciona colunas de variação (preço/vendas) com base nos prefixos declarados."""
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in VARIATION_PREFIXES):
            cols.append(c)
    return cols

def _model_feature_candidates(df: pd.DataFrame) -> list[str]:
    """
    Seleciona colunas candidatas a 'features de entrada' para cálculo de have_nulls:
    - Numéricas
    - EXCLUINDO regionais (não entram no modelo)
    - EXCLUINDO chaves/targets temporais listados em EXCLUDE_FROM_HAVE_NULLS
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [
        c for c in num_cols
        if c not in EXCLUDE_FROM_HAVE_NULLS and not _is_regional(c)
    ]
    return feat_cols

# -----------------------------------------------------------------------------
# Núcleo da imputação
# -----------------------------------------------------------------------------
def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a política de imputação:
      - Variações de preço e vendas (prefixos em VARIATION_PREFIXES): fillna(0)
      - Regionais, lags e rollings: manter NaN
      - Cria 'have_nulls' após imputação, olhando apenas para features de ENTRADA
        (exclui regionais e colunas de exclusão explícita).

    Compatível com bases agregadas sem categoria e com qualquer coluna temporal
    (order_week, order_date, etc.).
    """
    df = df.copy()

    # 1) Imputação de 0 nas variações (preço/vendas)
    var_cols = _variation_columns(df)
    if var_cols:
        before = int(df[var_cols].isna().sum().sum())
        df[var_cols] = df[var_cols].fillna(0.0)
        after = int(df[var_cols].isna().sum().sum())
        LOGGER.info("Variações imputadas em 0: %d faltantes -> %d.", before, after)
    else:
        LOGGER.info("Nenhuma coluna de variação encontrada (price/qty/sales).")

    # 2) NÃO imputar regionais, lags e rollings (mantém NaN)
    #    Nada a fazer aqui: a ausência é intencional.

    # 3) Gerar coluna have_nulls após imputações, apenas sobre features de ENTRADA
    feature_cols = _model_feature_candidates(df)
    if not feature_cols:
        # Se nada foi selecionado (cenário raro), caímos para numéricas gerais (menos arriscado)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c not in EXCLUDE_FROM_HAVE_NULLS]

    df["have_nulls"] = df[feature_cols].isna().any(axis=1).astype(np.uint8)

    return df

# -----------------------------------------------------------------------------
# Pipeline: leitura -> imputação -> escrita
# -----------------------------------------------------------------------------
def build_imputed_dataset(
    interim_dir: Optional[str | os.PathLike] = None,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    project_dir: Optional[str | os.PathLike] = None,
) -> Path:
    """
    1) Lê o parquet agregado (saída do build_features / aggregation).
       Pode ser semanal (order_week) ou diário (order_date), não importa.
    2) Aplica imputação conforme regras definidas.
    3) Salva parquet imputado em data/interim/.
    """
    proj_dir = _as_project_dir(project_dir)
    interim = _as_interim_dir(interim_dir, proj_dir)
    Path(interim).mkdir(parents=True, exist_ok=True)

    in_path = Path(interim).joinpath(input_name).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {in_path}")

    LOGGER.info("Lendo agregado temporal: %s", in_path)
    df = pd.read_parquet(in_path)

    # Diagnóstico pré-imputação
    na_before = int(df.isna().sum().sum())
    LOGGER.info("Faltantes totais (pré): %d", na_before)

    df_imp = impute_nulls(df)

    # Diagnóstico pós-imputação
    na_after = int(df_imp.isna().sum().sum())
    LOGGER.info("Faltantes totais (pós): %d", na_after)
    LOGGER.info("Proporção de linhas com have_nulls=1: %.2f%%", 100 * df_imp["have_nulls"].mean())

    out_path = Path(interim).joinpath(output_name).resolve()
    df_imp.to_parquet(out_path, index=False)
    LOGGER.info("Parquet imputado salvo em: %s (linhas=%d, colunas=%d)", out_path, len(df_imp), df_imp.shape[1])

    return out_path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Imputa valores nulos nas features agregadas (Olist).")
    p.add_argument("--interim", dest="interim", default=None, help="Diretório /data/interim")
    p.add_argument("--input", dest="input", default=DEFAULT_INPUT_NAME, help="Nome do parquet de entrada (agregado)")
    p.add_argument("--output", dest="output", default=DEFAULT_OUTPUT_NAME, help="Nome do parquet de saída (imputado)")
    p.add_argument("--project", dest="project", default=None, help="Diretório raiz do projeto (opcional)")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    build_imputed_dataset(
        interim_dir=args.interim,
        input_name=args.input,
        output_name=args.output,
        project_dir=args.project,
    )

if __name__ == "__main__":
    main()