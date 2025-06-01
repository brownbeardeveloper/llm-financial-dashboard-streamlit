import pandas as pd
import numpy as np


def calculate_percentage_change(data: pd.Series) -> pd.Series:
    """Calculate percentage change from first valid value."""
    first_valid_idx = data.first_valid_index()
    if first_valid_idx is None:
        return pd.Series([np.nan] * len(data), index=data.index)

    first_valid_value = data.loc[first_valid_idx]
    return ((data / first_valid_value) - 1) * 100
