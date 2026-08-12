"""Reusable quantitative tools for the multi-agent workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    annualized_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float
    risk_level: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _mock_prices(periods: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic prices so the project can run without a network."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    returns = rng.normal(loc=0.00035, scale=0.014, size=periods)
    prices = 180 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Close": prices}, index=dates)


def get_stock_data(ticker: str, period: str = "1y", use_mock_data: bool = False) -> pd.DataFrame:
    """Fetch historical close prices or use deterministic mock data."""
    if use_mock_data:
        return _mock_prices()

    try:
        import yfinance as yf

        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError(f"No market data returned for {ticker}.")
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.DataFrame({"Close": close.dropna()})
    except ImportError as exc:
        raise RuntimeError("Install yfinance to fetch live data: pip install yfinance") from exc


def assess_risk_metrics(prices: pd.Series, risk_free_rate: float = 0.0) -> RiskMetrics:
    """Calculate annualized volatility, drawdown, Sharpe, and historical VaR."""
    prices = pd.Series(prices, dtype=float).dropna()
    if len(prices) < 30:
        raise ValueError("At least 30 observations are required for risk analysis.")

    returns = prices.pct_change().dropna()
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    max_drawdown = float(drawdown.min())
    annual_return = float(returns.mean() * 252)
    sharpe = float((annual_return - risk_free_rate) / volatility) if volatility else 0.0
    var_95 = float(np.quantile(returns, 0.05))

    if volatility < 0.20 and max_drawdown > -0.20:
        level = "Low"
    elif volatility < 0.35 and max_drawdown > -0.35:
        level = "Moderate"
    else:
        level = "High"

    return RiskMetrics(volatility, max_drawdown, sharpe, var_95, level)


def format_financial_report(
    ticker: str,
    research: dict[str, Any],
    risk: RiskMetrics | dict[str, Any],
) -> str:
    """Convert structured agent results into a readable Markdown report."""
    risk_data = risk.to_dict() if isinstance(risk, RiskMetrics) else risk
    return f"""# Financial Intelligence Report: {ticker.upper()}

## Research Summary
- **Latest Close:** ${research['latest_price']:.2f}
- **Period Return:** {research['period_return']:.2%}
- **50-Day Moving Average:** ${research['moving_average_50']:.2f}
- **Annualized Volatility:** {research['annualized_volatility']:.2%}

## Risk Assessment
- **Maximum Drawdown:** {risk_data['max_drawdown']:.2%}
- **Sharpe Ratio:** {risk_data['sharpe_ratio']:.2f}
- **95% Historical VaR:** {risk_data['var_95']:.2%}
- **Risk Level:** **{risk_data['risk_level']}**

## Interpretation
The report summarizes historical market behavior and quantitative risk metrics. Historical
performance does not guarantee future results. This system does not provide investment advice.
"""
