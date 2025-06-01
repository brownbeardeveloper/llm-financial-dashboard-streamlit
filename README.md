# Finance Dashboard

This project presents a finance dashboard designed to simplify early-stage decision-making when evaluating whether a stock is worth investigating further. The application uses real-time financial data from the Yahoo Finance API, integrates OpenAI's LLM for automated insights, and is built with a Python-based Streamlit frontend.

## Demo

![Dashboard Demo](dashboard.gif)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Features

- Real-time stock data
- Technical indicators: SMA and RSI
- AI-generated summaries
- Interactive charts
- Benchmark comparison (e.g., S&P 500)

## Usage

1. Select one or more stocks from the sidebar
2. Choose a time period and technical indicators
3. Visualize trends, KPIs, and comparisons
4. Enter your OpenAI API key to receive AI-powered insights

```
project/
├── app.py                       # Main Streamlit application
├── utils.py                     # Helper functions
├── requirements.txt             # Python dependencies
├── data.ipynb                   # Data analysis notebook
├── classes/                     # Core modules
│   ├── data_loader.py           # Yahoo Finance data fetching
│   ├── technical_indicators.py  # SMA, RSI calculations
│   ├── openai_chat.py           # AI analysis integration
│   └── custom_exceptions.py     # Custom exception classes
└── README.md
```


## Developer's Note

The core of this project revolves around three main classes. Most development time was dedicated to making these classes robust and scalable, enabling them to be reused in other projects. I've tested and iterated on these components using a Jupyter notebook for easier debugging and experimentation.

The file data.ipynb serves as a development sandbox where I explored Yahoo Finance API data structures and tested key functionalities prior to integrating them into the main application.

Regarding the frontend, while I am comfortable with Streamlit and have used it alongside Pandas in previous data projects, I chose to prioritize the backend logic and prompt engineering for the LLM. For this phase, I collaborated with Claude 4 Sonnet to refine the user interface efficiently.

From a financial analysis perspective, the app pulls key performance indicators (KPIs) such as PE ratio, ROE, profit margin, current ratio, and beta from Yahoo Finance. It also calculates technical indicators like SMA and RSI. Users can select multiple stocks and benchmark them against an index like the S&P 500. These metrics are passed to the LLM, which analyzes and summarizes trends on a per-stock basis.

To allow others, such as my investment club peers, to use the app, I included a text input field where users can enter their OpenAI API key. The app does not store any keys or user data, but I strongly recommend removing your key from the OpenAI dashboard after use as a general best practice.
