# src/features/build.py
"""
Criação de features de séries temporais (lags e médias móveis) e
features de calendário (incluindo feriados BR) sobre a base semanal
agregada por categoria (saída do aggregation.py).

Regras importantes:
- Não há look-ahead: usamos shift() e rolling() com min_periods=window.
- As operações de lags/rollings são aplicadas por 'product_category_name'
  para respeitar as fronteiras de cada série.
- É esperado que surjam NaNs nas primeiras semanas (burn-in).
- As features de calendário são determinísticas e não usam dados futuros.

Saída padrão:
- data/interim/olist_weekly_agg_withlags.parquet
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feriados nacionais (biblioteca 'holidays'). Mantemos fallback elegante.
# ---------------------------------------------------------------------------
try:
    import holidays  # type: ignore
    _HAS_HOLIDAYS = True
except Exception:  # pragma: no cover
    holidays = None
    _HAS_HOLIDAYS = False

# ---------------------------------------------------------------------------
# Importa paths do loader com fallback robusto (igual espírito do aggregation)
# ---------------------------------------------------------------------------
try:
    # ao rodar como pacote (src/)
    from ..data.loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
    from ..data.loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
except Exception:
    try:
        # ao rodar a partir da raiz do repo / notebooks
        from src.data.loader import PROJECT_DIR as _PROJECT_DIR  # type: ignore
        from src.data.loader import INTERIM_DIR as _INTERIM_DIR  # type: ignore
    except Exception:
        _PROJECT_DIR = None
        _INTERIM_DIR = None

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
# Constantes / Parâmetros padrão
# ---------------------------------------------------------------------------
DEFAULT_INPUT_NAME = "olist_weekly_agg.parquet"
DEFAULT_OUTPUT_NAME = "olist_weekly_agg_withlags.parquet"

# Colunas base para lags/rollings — ajuste conforme necessidade
# Sempre garanta que existam no parquet de entrada.
LAG_COLS = [
    "sales_qty",                       # alvo (demanda)
    "price_var_w1_point_mean",         # variação semanal ponderada (produto→categoria)
    "price_var_w1_smooth_mean",        # variação semanal suavizada ponderada
    "price_var_m4_vs_prev4_mean",      # variação "mês móvel vs mês móvel anterior" ponderada
    # tempos ponderados por vendas regionais vindos do aggregation
    "approval_time_hours_weighted",
    "delivery_diff_estimated_weighted",
    "est_delivery_lead_days_weighted",
]

LAGS = [1, 2, 4, 8]    # semanas

ROLL_COLS = [
    "sales_qty",
    "price_var_w1_point_mean",
    "price_var_w1_smooth_mean",
    "price_var_m4_vs_prev4_mean",
    "approval_time_hours_weighted",
    "delivery_diff_estimated_weighted",
    "est_delivery_lead_days_weighted",
]

ROLL_WINDOWS = [4, 8]  # semanas

# ---------------------------------------------------------------------------
# Helpers de caminho e validação
# ---------------------------------------------------------------------------
def _as_project_dir(explicit: Optional[str | os.PathLike] = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if _PROJECT_DIR:
        return Path(_PROJECT_DIR).resolve()
    # fallback: src/features/build.py -> ... -> raiz do projeto
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

# ---------------------------------------------------------------------------
# Builders de lags e rollings
# ---------------------------------------------------------------------------
def _add_lags_per_group(g: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Aplica lags por grupo, ordenando por 'order_week'."""
    g = g.sort_values("order_week")
    for c in cols:
        if c not in g.columns:
            LOGGER.warning("Coluna para lag ausente e será ignorada: %s", c)
            continue
        for L in lags:
            g[f"{c}_lag{L}"] = g[c].shift(L)
    return g

def _add_rollings_per_group(g: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Aplica médias/desvios móveis CAUSAIS por grupo, ordenando por 'order_week'.

    Observação importante:
    - Para evitar look-ahead, calculamos o rolling sobre a série SHIFTADA em 1 (t-1, t-2, ...).
      Assim, o valor em t usa exclusivamente o passado.
    """
    g = g.sort_values("order_week")
    for c in cols:
        if c not in g.columns:
            LOGGER.warning("Coluna para rolling ausente e será ignorada: %s", c)
            continue
        s = g[c].shift(1)  # <- CAUSAL: exclui o valor da semana atual (t)
        for w in windows:
            r = s.rolling(window=w, min_periods=w)
            g[f"{c}_roll{w}_mean"] = r.mean()
            g[f"{c}_roll{w}_std"]  = r.std()
    return g

# ---------------------------------------------------------------------------
# Calendar features (determinísticas e sem vazamento)
# ---------------------------------------------------------------------------
def _week_of_year(dt: pd.Timestamp) -> int:
    # pandas >= 1.1: .isocalendar() retorna DataFrame com .week
    try:
        return int(dt.isocalendar().week)  # type: ignore[attr-defined]
    except Exception:
        # fallback: compatibilidade (deprecated em pandas recentes)
        return int(getattr(dt, "weekofyear", dt.week))

def _black_friday(date: pd.Timestamp) -> pd.Timestamp:
    """Última sexta-feira de novembro do ano de 'date'."""
    year = int(date.year)
    nov_start = pd.Timestamp(year=year, month=11, day=1, tz=getattr(date, "tz", None))
    # todas as sextas do mês de novembro e pegamos a última
    fridays = [d for d in pd.date_range(nov_start, nov_start + pd.offsets.MonthEnd(0), freq="W-FRI")]
    return fridays[-1]

def _cyber_monday(date: pd.Timestamp) -> pd.Timestamp:
    """Segunda-feira imediatamente após a Black Friday."""
    bf = _black_friday(date)
    return bf + pd.Timedelta(days=3)

def _holidays_span(min_date: pd.Timestamp, max_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Gera DataFrame com feriados nacionais BR no intervalo [min_date-30d, max_date+30d].
    Requer a lib 'holidays'. Se ausente, retorna None.
    """
    if not _HAS_HOLIDAYS:
        return None

    # amortecemos 30 dias para capturar bordas de semanas
    start = (min_date - pd.Timedelta(days=30)).normalize()
    end   = (max_date + pd.Timedelta(days=30)).normalize()

    years = list(range(start.year, end.year + 1))
    br = holidays.Brazil(years=years)  # type: ignore[attr-defined]

    # Cria lista de pares (date, name) apenas dentro do range desejado
    rows: List[Tuple[pd.Timestamp, str]] = []
    for d, name in br.items():
        dts = pd.Timestamp(d)
        if start <= dts <= end:
            rows.append((dts, str(name)))

    if not rows:
        return pd.DataFrame(columns=["date", "name"])

    hd = pd.DataFrame(rows, columns=["date", "name"]).sort_values("date").reset_index(drop=True)
    return hd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de calendário a partir de 'order_week' (datetime semanal).
    Não usa dados futuros e é 100% determinística.

    Colunas adicionadas:
      - year, quarter, month, weekofyear
      - month_sin, month_cos, week_sin, week_cos (codificação cíclica)
      - is_month_start, is_month_end, is_quarter_start
      - eventos de varejo: is_black_friday_week, is_cyber_monday_week, is_christmas_week
      - feriados BR (via 'holidays'): is_br_holiday_week, n_br_holidays_week
        (Flags específicas podem ser adicionadas futuramente se necessário.)
    """
    if "order_week" not in df.columns:
        raise KeyError("add_calendar_features: coluna 'order_week' ausente.")

    out = df.copy()
    ow = pd.to_datetime(out["order_week"])

    out["year"] = ow.dt.year
    out["quarter"] = ow.dt.quarter
    out["month"] = ow.dt.month

    # Semana ISO do ano
    out["weekofyear"] = ow.apply(_week_of_year)

    # Codificação cíclica (mês e semana do ano)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    out["week_sin"] = np.sin(2 * np.pi * (out["weekofyear"] - 1) / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * (out["weekofyear"] - 1) / 52.0)

    # Flags de borda de calendário
    out["is_month_start"] = ow.dt.is_month_start.astype(int)
    out["is_month_end"] = ow.dt.is_month_end.astype(int)
    out["is_quarter_start"] = ow.dt.is_quarter_start.astype(int)

    # Eventos de varejo (sem dependências externas)
    week_end = ow + pd.Timedelta(days=6)
    bf = ow.apply(_black_friday)
    cm = ow.apply(_cyber_monday)
    christmas = pd.to_datetime([pd.Timestamp(year=int(y), month=12, day=25) for y in out["year"]])

    out["is_black_friday_week"] = ((bf >= ow) & (bf <= week_end)).astype(int)
    out["is_cyber_monday_week"] = ((cm >= ow) & (cm <= week_end)).astype(int)
    out["is_christmas_week"] = ((christmas >= ow) & (christmas <= week_end)).astype(int)

    # Feriados nacionais BR (via 'holidays')
    if _HAS_HOLIDAYS:
        try:
            hd = _holidays_span(ow.min(), ow.max())
            if hd is None or hd.empty:
                out["is_br_holiday_week"] = 0
                out["n_br_holidays_week"] = 0
            else:
                # Para cada semana, conta quantos feriados caem entre [order_week, order_week+6]
                # Implementação vetorizada simples via merge-asof não é adequada por intervalos;
                # usamos um join cartesiano leve por ano para reduzir custo.
                # Estratégia: mapeamos por ano para cortar busca.
                hd["year"] = hd["date"].dt.year
                tmp = out[["order_week", "year"]].copy()
                tmp["week_end"] = week_end.values

                # Join por ano para reduzir espaço de comparação
                merged = tmp.merge(hd, on="year", how="left")
                in_week = (merged["date"] >= merged["order_week"]) & (merged["date"] <= merged["week_end"])

                # Conta por linha de semana
                counts = (
                    merged[in_week]
                    .groupby(merged.index.name if merged.index.name else merged.index)
                    .size()
                    .reindex(range(len(tmp)), fill_value=0)
                )

                out["n_br_holidays_week"] = counts.values
                out["is_br_holiday_week"] = (out["n_br_holidays_week"] > 0).astype(int)

        except Exception as e:  # pragma: no cover
            LOGGER.warning("Falha ao gerar feriados BR (holidays): %s", e)
            out["is_br_holiday_week"] = 0
            out["n_br_holidays_week"] = 0
    else:
        LOGGER.warning("Biblioteca 'holidays' não encontrada; flags de feriado serão 0.")
        out["is_br_holiday_week"] = 0
        out["n_br_holidays_week"] = 0

    return out

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def build_features(
    interim_dir: Optional[str | os.PathLike] = None,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    project_dir: Optional[str | os.PathLike] = None,
    lag_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    roll_cols: Optional[List[str]] = None,
    roll_windows: Optional[List[int]] = None,
) -> Path:
    """
    1) Lê o parquet agregado (aggregation.py).
    2) Cria lags e janelas móveis por categoria.
    3) Adiciona features de calendário (incl. feriados BR).
    4) Escreve parquet com as novas features.

    Parâmetros:
      - lag_cols/roll_cols: lista de colunas-alvo. Se None, usa defaults.
      - lags/roll_windows: lista de inteiros (semanas). Se None, usa defaults.
    """
    proj_dir = _as_project_dir(project_dir)
    interim = _as_interim_dir(interim_dir, proj_dir)
    Path(interim).mkdir(parents=True, exist_ok=True)

    in_path = Path(interim).joinpath(input_name).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet de entrada não encontrado: {in_path}")

    LOGGER.info("Lendo base semanal agregada: %s", in_path)
    df = pd.read_parquet(in_path)

    # Garantias mínimas
    _require_columns(df, ["product_category_name", "order_week"])
    # Normaliza tipos
    if not np.issubdtype(df["order_week"].dtype, np.datetime64):
        try:
            df["order_week"] = pd.to_datetime(df["order_week"])
        except Exception:
            raise TypeError("Coluna 'order_week' não é datetime e não pôde ser convertida.")

    # Parametrização efetiva
    lag_cols_eff  = lag_cols  if lag_cols  is not None else LAG_COLS
    lags_eff      = lags      if lags      is not None else LAGS
    roll_cols_eff = roll_cols if roll_cols is not None else ROLL_COLS
    roll_eff      = roll_windows if roll_windows is not None else ROLL_WINDOWS

    LOGGER.info("Aplicando LAGS: cols=%s | lags=%s", lag_cols_eff, lags_eff)
    df = (
        df.groupby("product_category_name", group_keys=False)
          .apply(lambda g: _add_lags_per_group(g, lag_cols_eff, lags_eff))
          .reset_index(drop=True)
    )

    LOGGER.info("Aplicando ROLLINGS (causais): cols=%s | windows=%s", roll_cols_eff, roll_eff)
    df = (
        df.groupby("product_category_name", group_keys=False)
          .apply(lambda g: _add_rollings_per_group(g, roll_cols_eff, roll_eff))
          .reset_index(drop=True)
    )

    LOGGER.info("Adicionando calendar features (incl. feriados BR) ...")
    df = add_calendar_features(df)

    out_path = Path(interim).joinpath(output_name).resolve()
    df.to_parquet(out_path, index=False)
    LOGGER.info("Features salvas: %s (linhas=%d, colunas=%d)", out_path, len(df), df.shape[1])

    return out_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gera lags, rollings e calendar features a partir da base semanal agregada.")
    p.add_argument("--interim", dest="interim", default=None, help="Diretório /data/interim")
    p.add_argument("--input", dest="input", default=DEFAULT_INPUT_NAME, help="Nome do parquet de entrada (aggregation)")
    p.add_argument("--output", dest="output", default=DEFAULT_OUTPUT_NAME, help="Nome do parquet de saída (features)")
    p.add_argument("--project", dest="project", default=None, help="Diretório raiz do projeto (opcional)")

    # Parâmetros opcionais para customizar via CLI (formato: 'col1,col2')
    p.add_argument("--lag-cols", dest="lag_cols", default=None, help="Colunas para lag, separadas por vírgula")
    p.add_argument("--lags", dest="lags", default=None, help="Lags em semanas, separados por vírgula (ex.: 1,2,4,8)")
    p.add_argument("--roll-cols", dest="roll_cols", default=None, help="Colunas para rolling, separadas por vírgula")
    p.add_argument("--roll-windows", dest="roll_windows", default=None, help="Janelas de rolling (ex.: 4,8)")
    return p

def _parse_list_arg(arg: Optional[str], cast=int) -> Optional[List]:
    if arg is None:
        return None
    parts = [a.strip() for a in arg.split(",") if a.strip() != ""]
    if cast is int:
        return [int(x) for x in parts]
    return parts  # strings

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    lag_cols = _parse_list_arg(args.lag_cols, cast=str)
    lags = _parse_list_arg(args.lags, cast=int)
    roll_cols = _parse_list_arg(args.roll_cols, cast=str)
    roll_windows = _parse_list_arg(args.roll_windows, cast=int)

    build_features(
        interim_dir=args.interim,
        input_name=args.input,
        output_name=args.output,
        project_dir=args.project,
        lag_cols=lag_cols,
        lags=lags,
        roll_cols=roll_cols,
        roll_windows=roll_windows,
    )

if __name__ == "__main__":
    main()
