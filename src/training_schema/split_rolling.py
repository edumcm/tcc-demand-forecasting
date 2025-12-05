# src/training_schema/split_rolling.py

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# ---------- Esquemas de treinamento ----------
def split_rolling(
    df: pd.DataFrame,
    date_col: str,
    first_train_end: pd.Timestamp,
    step: pd.Timedelta,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding window simples: a cada iteração,
    - treino = tudo até current_end
    - validação = (current_end, current_end + step]
    Avança current_end em 'step' a cada loop.
    """
    pairs = []
    current_end = first_train_end

    while True:
        val_end = current_end + step

        train = df[df[date_col] <= current_end].copy()
        valid = df[(df[date_col] > current_end) & (df[date_col] <= val_end)].copy()

        if valid.empty:
            break

        pairs.append((train, valid))
        current_end = val_end  # avança a janela

    return pairs
