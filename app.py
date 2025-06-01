import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from classes.data_loader import DataLoader
from classes.technical_indicators import TechnicalIndicators
from classes.openai_chat import StockAnalysisAI
from utils import calculate_percentage_change

# Page configuration
st.set_page_config(page_title="Financial Dashboard", page_icon="📈", layout="wide")


def main():
    st.title("📈 Financial Dashboard")

    with st.sidebar:
        st.header("Configuration")

        symbols = st.multiselect(
            "Companies",
            options=["BRK-B", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
            default=["BRK-B"],
        )

        period = st.selectbox(
            "Time Interval", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
        )

        sma_display = st.multiselect(
            "SMA Display",
            options=[5, 10, 20, 50, 100, 200],
            default=[50, 200],
            help="Choose which SMA lines to show in charts (all periods are calculated for AI analysis)",
        )

        rsi_period = st.slider("RSI Period", 10, 30, 14)

        benchmark = st.selectbox(
            "Benchmark",
            options=["^GSPC", "^IXIC", "^DJI"],
            format_func=lambda x: {
                "^GSPC": "S&P 500",
                "^IXIC": "NASDAQ",
                "^DJI": "Dow Jones",
            }[x],
        )

        enable_ai = st.checkbox("Enable AI", value=True)

        # OpenAI API Key
        if enable_ai:
            openai_key = st.text_input("OpenAI API Key", type="password")

    if symbols:
        data = load_data(symbols, benchmark, period, rsi_period)

        if data:
            tab1, tab2, tab3 = st.tabs(["📊 Charts", "🎯 KPIs", "🤖 AI Insights"])

            with tab1:
                display_charts(data, benchmark, sma_display)

            with tab2:
                display_kpis(data, symbols)

            with tab3:
                if enable_ai:
                    display_ai(data, symbols, benchmark, openai_key)
                else:
                    st.info("Enable AI in sidebar")


def load_data(symbols, benchmark, period, rsi_period):
    data = {}

    # Map display periods to extended periods for SMA calculation
    # Load extra data to ensure SMAs are available for the full display period
    extended_period_map = {
        "1mo": "1y",  # Load 1 year to calculate SMAs for 1 month display
        "3mo": "1y",  # Load 1 year to calculate SMAs for 3 months display
        "6mo": "2y",  # Load 2 years to calculate SMAs for 6 months display
        "1y": "2y",  # Load 2 years to calculate SMAs for 1 year display
        "2y": "5y",  # Load 5 years to calculate SMAs for 2 years display
        "5y": "max",  # Load max data for 5 years display
    }

    extended_period = extended_period_map.get(period, period)

    try:
        with st.spinner("Loading data..."):
            for symbol in symbols:
                # Load extended data for SMA calculation
                loader_extended = DataLoader(symbol, extended_period)
                stock_data_extended = loader_extended.data
                indicators = TechnicalIndicators(stock_data_extended)

                # Calculate all standard SMA periods automatically on extended data
                sma_periods = [5, 10, 20, 50, 100, 200]
                for sma_period in sma_periods:
                    stock_data_extended[f"sma_{sma_period}"] = indicators.sma(
                        sma_period
                    )

                stock_data_extended["rsi"] = indicators.rsi(window=rsi_period)

                # Now load the display period data and get the date range
                loader_display = DataLoader(symbol, period)
                stock_data_display = loader_display.data

                # Extract the display date range
                display_start = stock_data_display.index[0]
                display_end = stock_data_display.index[-1]

                # Trim extended data to display period while keeping calculated SMAs
                stock_data = stock_data_extended.loc[display_start:display_end].copy()

                # Add percentage change calculation
                stock_data["pct_change"] = calculate_percentage_change(
                    stock_data["close"]
                )

                data[symbol] = {"data": stock_data, "loader": loader_display}

            # Load benchmark data for display period only
            benchmark_loader = DataLoader(benchmark, period)
            benchmark_data = benchmark_loader.data
            benchmark_data["pct_change"] = calculate_percentage_change(
                benchmark_data["close"]
            )

            data["benchmark"] = {"data": benchmark_data, "symbol": benchmark}

        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def display_charts(data, benchmark, sma_display):
    benchmark_data = data["benchmark"]["data"]
    benchmark_symbol = data["benchmark"]["symbol"]

    # Convert benchmark symbol to friendly name
    benchmark_name = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones"}.get(
        benchmark_symbol, benchmark_symbol
    )

    st.subheader("Performance Comparison")

    fig = go.Figure()

    for symbol in data:
        if symbol != "benchmark":
            stock_data = data[symbol]["data"]

            # Add main stock performance line
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data["pct_change"],
                    name=symbol,
                    line=dict(width=3),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data["pct_change"],
            name=f"{benchmark_name} (Market Index)",
            line=dict(color="orange", width=1),
        )
    )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.5)

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add explanation note
    st.caption(
        f"📊 Chart shows percentage returns for your selected companies vs the {benchmark_name} market index (orange line)"
    )

    # Improved layout for multiple companies
    num_companies = len([s for s in data if s != "benchmark"])

    if num_companies <= 4:
        cols = st.columns(num_companies)
        for i, symbol in enumerate([s for s in data if s != "benchmark"]):
            with cols[i]:
                display_technical_chart(data[symbol]["data"], symbol, sma_display)
    else:
        # Use 2 rows for better readability when many companies
        companies = [s for s in data if s != "benchmark"]

        # First row - up to 4 companies
        cols1 = st.columns(min(4, len(companies)))
        for i in range(min(4, len(companies))):
            with cols1[i]:
                display_technical_chart(
                    data[companies[i]]["data"], companies[i], sma_display
                )

        # Second row - remaining companies
        if len(companies) > 4:
            remaining = companies[4:]
            cols2 = st.columns(len(remaining))
            for i, symbol in enumerate(remaining):
                with cols2[i]:
                    display_technical_chart(data[symbol]["data"], symbol, sma_display)


def display_technical_chart(stock_data, symbol, sma_display):
    st.subheader(f"{symbol} Technical")

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Price & SMA", "Volume", "RSI"),
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.3,
    )

    # Always green close price
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data["close"],
            name="Close",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
    )

    # Add all SMA lines with distinct colors
    sma_columns = [col for col in stock_data.columns if col.startswith("sma_")]
    sma_colors = [
        "red",
        "blue",
        "purple",
        "orange",
        "brown",
        "darkred",
    ]  # SMA-specific colors

    for i, sma_col in enumerate(sma_columns):
        sma_period = sma_col.split("_")[1]
        if int(sma_period) in sma_display:
            color = sma_colors[i % len(sma_colors)]
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data[sma_col],
                    name=f"SMA{sma_period}",
                    line=dict(color=color, width=1),
                ),
                row=1,
                col=1,
            )

    # Always green volume
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data["volume"],
            name="Volume",
            marker_color="green",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Always purple RSI
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data["rsi"],
            name="RSI",
            line=dict(color="purple", width=2),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Add y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def display_kpis(data, symbols):
    st.subheader("Key Performance Indicators")

    for symbol in symbols:
        st.markdown(f"### {symbol}")

        stock_data = data[symbol]["data"]
        kpi_summary = data[symbol]["loader"].get_kpi_summary()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            current_price = stock_data["close"].iloc[-1]
            period_return = stock_data["pct_change"].iloc[-1]
            st.metric("Price", f"${current_price:.2f}", f"{period_return:.2f}%")

        with col2:
            volatility = stock_data["close"].pct_change().std() * np.sqrt(252) * 100
            st.metric("Volatility", f"{volatility:.1f}%")

        with col3:
            current_rsi = stock_data["rsi"].iloc[-1]
            if not pd.isna(current_rsi):
                st.metric("RSI", f"{current_rsi:.1f}")

        with col4:
            # Get first available SMA period
            sma_columns = [col for col in stock_data.columns if col.startswith("sma_")]
            if sma_columns:
                first_sma_col = sma_columns[0]
                sma_period = first_sma_col.split("_")[1]
                current_sma = stock_data[first_sma_col].iloc[-1]
                st.metric(f"SMA{sma_period}", f"${current_sma:.2f}")
            else:
                st.metric("SMA", "N/A")

        if kpi_summary:
            col1, col2, col3, col4, col5 = st.columns(5)

            metrics = [
                ("P/E", "pe"),
                ("Profit Margin", "profit_margin"),
                ("ROE", "roe"),
                ("Current Ratio", "current_ratio"),
                ("Beta", "beta"),
            ]

            for i, (label, key) in enumerate(metrics):
                if key in kpi_summary:
                    value = kpi_summary[key]
                    if pd.notna(value):
                        if key in ["profit_margin", "roe"]:
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:.2f}"

                        with [col1, col2, col3, col4, col5][i]:
                            st.metric(label, formatted_value)


def display_ai(data, symbols, benchmark, openai_key):
    st.subheader("AI Investment Analysis")

    if not openai_key:
        st.warning("Enter OpenAI API key in sidebar")
        return

    try:
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        ai_analyzer = StockAnalysisAI()

        for symbol in symbols:
            st.markdown(f"### {symbol} Analysis")

            with st.spinner(f"Analyzing {symbol}..."):
                stock_data = data[symbol]["data"].copy()
                stock_data["ma50"] = stock_data["sma_50"]
                stock_data["ma200"] = stock_data["sma_200"]
                stock_data["rsi_14"] = stock_data["rsi"]

                benchmark_data = data["benchmark"]["data"]

                analysis = ai_analyzer.analyze_stock(
                    stock_data, benchmark_data, symbol, benchmark
                )

                st.markdown(
                    f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff;">
                    {analysis}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"AI Analysis failed: {e}")


if __name__ == "__main__":
    main()
