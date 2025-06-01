import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Simple class for computing technical indicators on stock data."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data.

        Args:
            data: DataFrame with OHLCV data from DataLoader (lowercase columns: open, high, low, close, volume)
        """
        if data.empty:
            raise ValueError("Data cannot be empty")

        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        self.data = data.copy()

    def sma(self, period: int, column: str = "close") -> pd.Series:
        """
        Calculate Simple Moving Average (SMA) for a specified period.

        Args:
            period (int): Number of periods over which to calculate SMA.
            column (str): Column name to calculate SMA (default: 'close').

        Returns:
            pd.Series: Computed SMA series.

        Raises:
            ValueError: If data isn't loaded or column isn't found.
        """
        if getattr(self, "data", None) is None:
            raise ValueError("Data is not loaded. Load data before computing SMA.")

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        return self.data[column].rolling(window=period, min_periods=period).mean()

    def rsi(self, data_column: str = "close", window: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) using Wilder's smoothing method.

        Args:
            data_column (str): Column name to calculate RSI on (default: 'close').
            window (int): Look-back period (default: 14).

        Returns:
            pd.Series: RSI values (0-100).
        """
        if data_column not in self.data.columns:
            raise ValueError(f"Column '{data_column}' not found in data")

        data = self.data[data_column]

        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gains = delta.where(delta >= 0, 0).fillna(0)
        losses = -delta.where(delta < 0, 0).fillna(0)

        # Initialize average arrays
        avg_gains = pd.Series(index=data.index, dtype=float)
        avg_losses = pd.Series(index=data.index, dtype=float)

        # Calculate initial averages and apply Wilder's smoothing
        if len(gains) >= window:
            first_avg_gain = gains.iloc[1 : window + 1].mean()
            first_avg_loss = losses.iloc[1 : window + 1].mean()

            avg_gains.iloc[window] = first_avg_gain
            avg_losses.iloc[window] = first_avg_loss

            # Apply Wilder's smoothing for remaining periods
            for i in range(window + 1, len(data)):
                avg_gains.iloc[i] = (
                    avg_gains.iloc[i - 1] * (window - 1) + gains.iloc[i]
                ) / window
                avg_losses.iloc[i] = (
                    avg_losses.iloc[i - 1] * (window - 1) + losses.iloc[i]
                ) / window

        # Calculate RS and RSI
        rs = avg_gains / avg_losses

        # Handle division by zero
        rs = rs.replace([np.inf, -np.inf], np.nan)

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Set RSI = 100 when avg_losses = 0
        rsi = rsi.fillna(100)

        # Set initial period to NaN (insufficient data)
        rsi.iloc[:window] = pd.NA

        return rsi
