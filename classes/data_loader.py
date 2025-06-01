import pandas as pd
import yfinance as yf
import numpy as np
from typing import Optional


class DataLoader:
    """
    Simple class for loading stock data with top 5 financial KPIs.

    Provides OHLCV price data and essential financial metrics:
    - P/E Ratio (valuation)
    - Profit Margin (profitability)
    - ROE (shareholder returns)
    - Current Ratio (liquidity)
    - Beta (market risk)
    """

    VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

    def __init__(self, symbol: str, period: str = "1y"):
        """
        Initialize DataLoader for a specific stock symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Time period for data ('1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Raises:
            ValueError: If symbol is empty or period is invalid
        """
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty.")

        if period not in self.VALID_PERIODS:
            raise ValueError("Invalid period.")

        self.symbol = symbol.upper().strip()
        self.period = period
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Get stock data with top 5 financial KPIs.

        Returns:
            DataFrame with columns: open, high, low, close, volume, pe, profit_margin, roe, current_ratio, beta

        Raises:
            ValueError: If data or company information unavailable
            Other exceptions from yfinance library may bubble up naturally
        """
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> pd.DataFrame:
        """
        Load historical data with top 5 most important financial KPIs.

        Returns:
            DataFrame with OHLCV data and financial KPIs

        Raises:
            ValueError: If no data found or company info unavailable
            Other exceptions from yfinance library bubble up naturally
        """
        try:
            ticker = yf.Ticker(self.symbol)
            history_data = ticker.history(period=self.period)

            if history_data.empty:
                raise ValueError(f"No data found for '{self.symbol}'")

            info = ticker.info

            if not info:
                raise ValueError(f"Company KPIs unavailable for '{self.symbol}'")

            # Create DataFrame
            data = pd.DataFrame()
            data["date"] = history_data.index.date
            data["open"] = history_data["Open"].values
            data["high"] = history_data["High"].values
            data["low"] = history_data["Low"].values
            data["close"] = history_data["Close"].values
            data["volume"] = history_data["Volume"].values

            # Add KPIs
            data["pe"] = info.get("trailingPE", np.nan)
            data["profit_margin"] = info.get("profitMargins", np.nan)
            data["roe"] = info.get("returnOnEquity", np.nan)
            data["current_ratio"] = info.get("currentRatio", np.nan)
            data["beta"] = info.get("beta", np.nan)

            # Set date as index
            data.set_index("date", inplace=True)
            return data

        except ValueError as e:
            raise e

    def get_kpi_summary(self) -> dict:
        """
        Get summary of the top 5 KPIs (non-NaN values only).

        Returns:
            Dictionary with available KPI values
        """
        latest = self.data.iloc[-1]
        summary = {}
        kpis = ["pe", "profit_margin", "roe", "current_ratio", "beta"]

        for kpi in kpis:
            value = latest[kpi]
            if not pd.isna(value):
                summary[kpi] = value

        return summary

    def refresh(self) -> None:
        """Clear cached data to force reload."""
        self._data = None
