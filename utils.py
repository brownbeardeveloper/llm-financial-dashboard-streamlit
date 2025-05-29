import pandas as pd


def calculate_percentage_change(data: pd.Series) -> pd.Series:
    """Calculate percentage change from first day."""
    return ((data / data.iloc[0]) - 1) * 100
