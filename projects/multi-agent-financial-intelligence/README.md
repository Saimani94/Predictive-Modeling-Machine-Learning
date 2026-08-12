# Multi-Agent Financial Intelligence System

A sequentially orchestrated multi-agent pipeline for automated equity research, quantitative risk analysis, and financial reporting.

## What it demonstrates

- Coordinator-driven multi-agent orchestration
- Specialized Research, Risk, and Reporting agents
- Reusable Python tooling
- Live market data through `yfinance`
- Deterministic mock mode for reproducibility
- Automated tests for tools and the end-to-end workflow

## Architecture

```text
Ticker Request
     |
     v
Coordinator Agent
     |
     +--> Research Agent --> market indicators
     |
     +--> Risk Agent -----> volatility / drawdown / Sharpe / VaR
     |
     +--> Reporting Agent -> Markdown financial intelligence report
```

## Run

From this directory:

```bash
pip install -r requirements.txt
python main.py --ticker AAPL --period 1y --mock
```

For live data:

```bash
python main.py --ticker AAPL --period 1y
```

## Test

```bash
pytest -q
```

## Core tools

`get_stock_data`, `assess_risk_metrics`, and `format_financial_report` isolate data access, quantitative analysis, and report generation from orchestration logic.

## Portfolio note

This project is designed to demonstrate software architecture and financial analytics. It does not execute trades or provide personalized investment advice.
