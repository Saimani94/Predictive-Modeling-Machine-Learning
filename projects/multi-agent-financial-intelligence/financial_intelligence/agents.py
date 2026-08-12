"""Specialized agents used by the sequential coordinator."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .tools import RiskMetrics, assess_risk_metrics, format_financial_report


class ResearchAgent:
    """Analyzes market history and produces structured research findings."""

    name = "Research Agent"

    def run(self, ticker: str, prices: pd.DataFrame) -> dict[str, Any]:
        close = prices["Close"].astype(float).dropna()
        returns = close.pct_change().dropna()
        return {
            "ticker": ticker.upper(),
            "latest_price": float(close.iloc[-1]),
            "period_return": float(close.iloc[-1] / close.iloc[0] - 1),
            "moving_average_50": float(close.tail(50).mean()),
            "annualized_volatility": float(returns.std(ddof=1) * 252**0.5),
            "observations": int(len(close)),
        }


class RiskAgent:
    """Calculates quantitative risk metrics from research data."""

    name = "Risk Agent"

    def run(self, prices: pd.DataFrame) -> RiskMetrics:
        return assess_risk_metrics(prices["Close"])


class ReportingAgent:
    """Produces the final structured financial intelligence report."""

    name = "Reporting Agent"

    def run(self, ticker: str, research: dict[str, Any], risk: RiskMetrics) -> str:
        return format_financial_report(ticker, research, risk)
