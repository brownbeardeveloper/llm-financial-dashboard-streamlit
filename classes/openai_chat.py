import time
import os
from typing import List, Dict, Optional, Generator, Union
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError
import pandas as pd
from utils import calculate_percentage_change
from .custom_exceptions import (
    ChatError,
    MissingOpenAIKeyError,
    MessageValidationError,
    OpenAIErrorParser,
)


class StockAnalysisAI:
    """OpenAI chat client optimized for stock analysis and financial data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_retries: int = 2,
        retry_delay: float = 0.5,  # seconds
    ):
        """
        Initialize the stock analysis AI client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (0.1 for consistent analysis)
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds

        Raises:
            MissingOpenAIKeyError: If no API key is provided
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise MissingOpenAIKeyError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send chat messages to OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            String response or generator for streaming

        Raises:
            ChatError: If OpenAI API fails after retries
            MessageValidationError: If message format is invalid
        """
        # Validate messages
        self._validate_messages(messages)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except (RateLimitError, APITimeoutError, APIError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

                # Convert to user-friendly error message
                user_error = OpenAIErrorParser.get_user_friendly_message(e)
                raise ChatError(user_error)

    def analyze_stock(
        self,
        stock: pd.DataFrame,
        index: pd.DataFrame,
        stock_symbol: str,
        index_symbol: str,
    ) -> str:
        """
        Analyze a stock versus its benchmark using DataFrames and KPIs.
        Generates a concise, institution-grade investment summary.

        Args:
            stock: DataFrame with stock data including close, volume, indicators
            index: DataFrame with benchmark index data
            stock_symbol: Stock ticker symbol
            index_symbol: Index ticker symbol

        Returns:
            Institution-grade investment summary

        Raises:
            ChatError: If OpenAI API fails
            MessageValidationError: If message format is invalid
            MissingOpenAIKeyError: If API key is missing
        """
        # Calculate period percentage changes
        stock_change_pct = calculate_percentage_change(stock["close"]).iloc[-1]
        index_change_pct = calculate_percentage_change(index["close"]).iloc[-1]

        # KPI keys to extract and show
        kpi_keys = ["pe", "profit_margin", "roe", "current_ratio", "beta"]
        latest_kpis = {
            key: stock[key].iloc[-1] if key in stock.columns else None
            for key in kpi_keys
        }

        # Format profit margin and ROE as percentages for display
        if latest_kpis["profit_margin"] is not None:
            profit_margin_pct = latest_kpis["profit_margin"] * 100
        else:
            profit_margin_pct = None

        if latest_kpis["roe"] is not None:
            roe_pct = latest_kpis["roe"] * 100
        else:
            roe_pct = None

        prompt = f"""
        Write a concise investment summary (3-5 sentences) for the stock {stock_symbol}, suitable for institutional portfolio review.

        Period percentage change:
        - {stock_symbol}: {stock_change_pct:+.2f}%
        - {index_symbol}: {index_change_pct:+.2f}%

        Key latest KPIs:
        - P/E: {latest_kpis["pe"]}
        - Profit Margin: {profit_margin_pct}%
        - ROE: {roe_pct}%
        - Current Ratio: {latest_kpis["current_ratio"]}
        - Beta: {latest_kpis["beta"]}

        Big Data:
        - Closing Prices (period): {stock["close"].tolist()}
        - RSI values: {stock["rsi_14"].tolist() if "rsi_14" in stock.columns else "N/A"}
        - MA50: {stock["ma50"].tolist() if "ma50" in stock.columns else "N/A"}
        - MA200: {stock["ma200"].tolist() if "ma200" in stock.columns else "N/A"}
        - Volume: {stock["volume"].tolist() if "volume" in stock.columns else "N/A"}

        {index_symbol} Closing Prices (period): {index["close"].tolist()}

        Instructions:
        1.	Begin with the stock's recent price trend and volatility.
        2.	Compare its performance to {index_symbol} in percentage terms.
        3.	Evaluate RSI: note if oversold (<30), overbought (>70), or near boundaries.
        4.	Comment on MA50 vs MA200: note position and crossovers (death cross if MA50 < MA200).
        5.	If price drops below MA200, interpret as negative trend break.
        6.	If both (death cross + price below MA200), treat as strong bearish unless RSI contradicts.
        7.	Analyze recent trading volume: note significant spikes or drops and the implications for trend strength.
        8.	Analyze all available KPIs: valuation (P/E), profitability (Profit Margin, ROE), liquidity (Current Ratio), risk (Beta), sector context, etc.

        Base assessment strictly on data above. No speculation.
    """

        return self.quick_analysis(prompt)

    def quick_analysis(self, prompt: str) -> str:
        """
        Quick analysis using a single prompt.

        Args:
            prompt: Analysis prompt with data and instructions

        Returns:
            AI-generated analysis

        Raises:
            ChatError: If OpenAI API fails
            MessageValidationError: If message format is invalid
            MissingOpenAIKeyError: If API key is missing
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Wall Street financial analyst. "
                    "Provide institution-grade, concise, data-driven investment summaries suitable for portfolio review. "
                    "Never speculate beyond the user's provided data."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages)

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format."""
        if not messages:
            raise MessageValidationError("Messages list cannot be empty")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise MessageValidationError(f"Message {i} must be a dictionary")

            if "role" not in msg or "content" not in msg:
                raise MessageValidationError(
                    f"Message {i} must have 'role' and 'content' keys"
                )

            if msg["role"] not in ["system", "user", "assistant"]:
                raise MessageValidationError(
                    f"Message {i} role must be 'system', 'user', or 'assistant'"
                )
