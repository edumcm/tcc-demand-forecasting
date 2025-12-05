# src/features/build.py
"""
Criação de features de séries temporais (lags e médias móveis) e
features de calendário (incluindo feriados BR) sobre a base temporal
agregada (semanal OU diária), já consolidada no nível global
(sem segmentação por categoria).

Regras importantes:
- Não há look-ahead: usamos shift() e rolling() causais.
- As operações de lags/rollings são aplicadas sobre a série global
  (uma linha por período: semana ou dia).
- As features de calendário são determinísticas e não usam dados futuros.

Saída padrão (semanal):
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
# Importa paths do loader com fallback robusto
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
LOGGER.propagate = False

# ---------------------------------------------------------------------------
# Constantes / Parâmetros padrão
# ---------------------------------------------------------------------------
DEFAULT_INPUT_NAME = "olist_weekly_agg.parquet"
DEFAULT_OUTPUT_NAME = "olist_weekly_agg_withlags.parquet"

# Colunas base para lags/rollings — ajuste conforme necessidade.
LAG_COLS = [
    "sales_qty",
    "price_var_w1_point_mean",
    "price_var_w1_smooth_mean",
    "price_var_m4_vs_prev4_mean",
    "approval_time_hours_weighted",
    "delivery_diff_estimated_weighted",
    "est_delivery_lead_days_weighted",
]

# Interpretação: períodos (se semana -> semanas; se dia -> dias)
LAGS = [1, 2, 4, 8]

ROLL_COLS = [
    "sales_qty",
    "price_var_w1_point_mean",
    "price_var_w1_smooth_mean",
    "price_var_m4_vs_prev4_mean",
    "approval_time_hours_weighted",
    "delivery_diff_estimated_weighted",
    "est_delivery_lead_days_weighted",
]

# Interpretação: janelas em nº de períodos (semanas ou dias)
ROLL_WINDOWS = [2, 3, 4]

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

def _detect_time_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    """
    Detecta a coluna temporal a ser usada:
      - se explicit for fornecida, valida e retorna;
      - senão, tenta 'order_week' (semanal) depois 'order_date' (diária).
    """
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Coluna temporal '{explicit}' não encontrada no DataFrame.")
        return explicit
    if "order_week" in df.columns:
        return "order_week"
    if "order_date" in df.columns:
        return "order_date"
    raise KeyError("Nenhuma coluna temporal encontrada. Esperado 'order_week' ou 'order_date'.")

# ---------------------------------------------------------------------------
# Builders de lags e rollings (série global, sem categoria)
# ---------------------------------------------------------------------------
def _add_lags_global(df: pd.DataFrame, time_col: str, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Aplica lags na série temporal global, ordenando por time_col.

    Obs.: assumimos que o parquet de entrada está no nível temporal
    (uma linha por período) e *não* possui mais segmentação por categoria.
    """
    df = df.sort_values(time_col).copy()
    for c in cols:
        if c not in df.columns:
            LOGGER.warning("Coluna para lag ausente e será ignorada: %s", c)
            continue
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def _add_rollings_global(df: pd.DataFrame, time_col: str, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Aplica médias/desvios móveis CAUSAIS na série global, ordenando por time_col.

    Para evitar look-ahead, calculamos o rolling sobre a série SHIFTADA em 1 (t-1, t-2, ...).
    """
    df = df.sort_values(time_col).copy()
    for c in cols:
        if c not in df.columns:
            LOGGER.warning("Coluna para rolling ausente e será ignorada: %s", c)
            continue
        s = df[c].shift(1)  # <- CAUSAL: exclui o valor do período atual (t)
        for w in windows:
            r = s.rolling(window=w, min_periods=w)
            df[f"{c}_roll{w}_mean"] = r.mean()
            df[f"{c}_roll{w}_std"] = r.std()
    return df

# ---------------------------------------------------------------------------
# Calendar features (determinísticas e sem vazamento) – semanal ou diária
# ---------------------------------------------------------------------------
def _week_of_year(dt: pd.Timestamp) -> int:
    # pandas >= 1.1: .isocalendar() retorna DataFrame com .week
    try:
        return int(dt.isocalendar().week)  # type: ignore[attr-defined]
    except Exception:
        return int(getattr(dt, "weekofyear", dt.week))

def _black_friday(date: pd.Timestamp) -> pd.Timestamp:
    """Última sexta-feira de novembro do ano de 'date'."""
    year = int(date.year)
    nov_start = pd.Timestamp(year=year, month=11, day=1, tz=getattr(date, "tz", None))
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

    start = (min_date - pd.Timedelta(days=30)).normalize()
    end = (max_date + pd.Timedelta(days=30)).normalize()
    years = list(range(start.year, end.year + 1))
    br = holidays.Brazil(years=years)  # type: ignore[attr-defined]

    rows: List[Tuple[pd.Timestamp, str]] = []
    for d, name in br.items():
        dts = pd.Timestamp(d)
        if start <= dts <= end:
            rows.append((dts, str(name)))

    if not rows:
        return pd.DataFrame(columns=["date", "name"])

    hd = pd.DataFrame(rows, columns=["date", "name"]).sort_values("date").reset_index(drop=True)
    return hd

def _infer_freq_from_dates(dates: pd.Series) -> str:
    """
    Inferência simples da frequência temporal:
      - Se mediana do delta em dias >= 6 -> assume semanal ('W').
      - Caso contrário -> assume diária ('D').
    """
    uniques = np.sort(pd.to_datetime(dates).unique())
    if len(uniques) < 2:
        return "D"
    deltas = np.diff(uniques.astype("datetime64[D]").astype(int))
    median_delta = np.median(deltas)
    return "W" if median_delta >= 6 else "D"

def add_calendar_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Cria features de calendário a partir da coluna temporal (semanal ou diária).
    Não usa dados futuros e é 100% determinística.

    Colunas adicionadas:
      - year, quarter, month, weekofyear
      - month_sin, month_cos, week_sin, week_cos (codificação cíclica)
      - is_month_start, is_month_end, is_quarter_start
      - eventos de varejo: is_black_friday_week, is_cyber_monday_week, is_christmas_week
      - feriados BR (via 'holidays'): is_br_holiday_week, n_br_holidays_week

    Observação:
    - Mesmo para agregação diária, mantemos os nomes *_week por compatibilidade,
      mas eles significam "este dia está dentro da semana de Black Friday / Natal / etc.".
    """
    if time_col not in df.columns:
        raise KeyError(f"add_calendar_features: coluna temporal '{time_col}' ausente.")

    out = df.copy()
    t = pd.to_datetime(out[time_col])

    # Infere frequência
    freq = _infer_freq_from_dates(t)
    LOGGER.info("add_calendar_features: inferida frequência '%s' para coluna temporal '%s'.", freq, time_col)

    out["year"] = t.dt.year
    out["quarter"] = t.dt.quarter
    out["month"] = t.dt.month

    out["weekofyear"] = t.apply(_week_of_year)

    # Codificação cíclica
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    out["week_sin"] = np.sin(2 * np.pi * (out["weekofyear"] - 1) / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * (out["weekofyear"] - 1) / 52.0)

    # Flags de borda de calendário (funcionam bem para dia ou semana)
    out["is_month_start"] = t.dt.is_month_start.astype(int)
    out["is_month_end"] = t.dt.is_month_end.astype(int)
    out["is_quarter_start"] = t.dt.is_quarter_start.astype(int)

    # Janela de período (para eventos): se semanal, 7 dias; se diário, apenas 1 dia.
    if freq == "W":
        period_start = t
        period_end = t + pd.Timedelta(days=6)
    else:
        period_start = t
        period_end = t

    bf = t.apply(_black_friday)
    cm = t.apply(_cyber_monday)
    christmas = pd.to_datetime([pd.Timestamp(year=int(y), month=12, day=25) for y in out["year"]])

    out["is_black_friday_week"] = ((bf >= period_start) & (bf <= period_end)).astype(int)
    out["is_cyber_monday_week"] = ((cm >= period_start) & (cm <= period_end)).astype(int)
    out["is_christmas_week"] = ((christmas >= period_start) & (christmas <= period_end)).astype(int)

    # Feriados nacionais BR
    if _HAS_HOLIDAYS:
        try:
            hd = _holidays_span(t.min(), t.max())
            if hd is None or hd.empty:
                out["is_br_holiday_week"] = 0
                out["n_br_holidays_week"] = 0
            else:
                hd["year"] = hd["date"].dt.year
                tmp = out[[time_col, "year"]].copy()
                tmp["period_start"] = period_start.values
                tmp["period_end"] = period_end.values

                merged = tmp.merge(hd, on="year", how="left")
                in_period = (merged["date"] >= merged["period_start"]) & (merged["date"] <= merged["period_end"])

                counts = (
                    merged[in_period]
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
    time_col: Optional[str] = None,
) -> Path:
    """
    1) Lê o parquet agregado (aggregation.py), que pode estar:
         - em nível semanal (coluna 'order_week'), ou
         - em nível diário (coluna 'order_date').
    2) Detecta a coluna temporal (ou usa 'time_col' se fornecida).
    3) Cria lags e janelas móveis na série temporal global.
    4) Adiciona features de calendário (incl. feriados BR).
    5) Escreve parquet com as novas features.

    Parâmetros:
      - lag_cols/roll_cols: lista de colunas-alvo. Se None, usa defaults.
      - lags/roll_windows: lista de inteiros (períodos). Se None, usa defaults.
      - time_col: nome da coluna temporal; se None, detecta automaticamente.
    """
    proj_dir = _as_project_dir(project_dir)
    interim = _as_interim_dir(interim_dir, proj_dir)
    Path(interim).mkdir(parents=True, exist_ok=True)

    in_path = Path(interim).joinpath(input_name).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet de entrada não encontrado: {in_path}")

    LOGGER.info("Lendo base agregada: %s", in_path)
    df = pd.read_parquet(in_path)

    # Detecta e valida coluna temporal
    time_col_eff = _detect_time_col(df, explicit=time_col)
    if not np.issubdtype(df[time_col_eff].dtype, np.datetime64):
        try:
            df[time_col_eff] = pd.to_datetime(df[time_col_eff])
        except Exception:
            raise TypeError(f"Coluna temporal '{time_col_eff}' não é datetime e não pôde ser convertida.")

    # Parametrização efetiva
    lag_cols_eff = lag_cols if lag_cols is not None else LAG_COLS
    lags_eff = lags if lags is not None else LAGS
    roll_cols_eff = roll_cols if roll_cols is not None else ROLL_COLS
    roll_eff = roll_windows if roll_windows is not None else ROLL_WINDOWS

    LOGGER.info(
        "Aplicando LAGS (série global, time_col=%s): cols=%s | lags=%s",
        time_col_eff, lag_cols_eff, lags_eff,
    )
    df = _add_lags_global(df, time_col_eff, lag_cols_eff, lags_eff)

    LOGGER.info(
        "Aplicando ROLLINGS (causais, série global, time_col=%s): cols=%s | windows=%s",
        time_col_eff, roll_cols_eff, roll_eff,
    )
    df = _add_rollings_global(df, time_col_eff, roll_cols_eff, roll_eff)

    LOGGER.info("Adicionando calendar features (incl. feriados BR) ...")
    df = add_calendar_features(df, time_col=time_col_eff)

    out_path = Path(interim).joinpath(output_name).resolve()
    df.to_parquet(out_path, index=False)
    LOGGER.info("Features salvas: %s (linhas=%d, colunas=%d)", out_path, len(df), df.shape[1])

    return out_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Gera lags, rollings e calendar features a partir da base temporal "
            "agregada (semanal ou diária, global)."
        )
    )
    p.add_argument("--interim", dest="interim", default=None, help="Diretório /data/interim")
    p.add_argument("--input", dest="input", default=DEFAULT_INPUT_NAME, help="Nome do parquet de entrada (aggregation)")
    p.add_argument("--output", dest="output", default=DEFAULT_OUTPUT_NAME, help="Nome do parquet de saída (features)")
    p.add_argument("--project", dest="project", default=None, help="Diretório raiz do projeto (opcional)")
    p.add_argument(
        "--time-col",
        dest="time_col",
        default=None,
        help="Nome da coluna temporal ('order_week' ou 'order_date'). Se omitido, detecta automaticamente.",
    )

    # Parâmetros opcionais para customizar via CLI (formato: 'col1,col2')
    p.add_argument("--lag-cols", dest="lag_cols", default=None, help="Colunas para lag, separadas por vírgula")
    p.add_argument("--lags", dest="lags", default=None, help="Lags em períodos, separados por vírgula (ex.: 1,2,4,8)")
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
        time_col=args.time_col,
    )

if __name__ == "__main__":
    main()
