"""Sequential orchestration layer for the financial intelligence system."""

from __future__ import annotations

from typing import Any

from .agents import ReportingAgent, ResearchAgent, RiskAgent
from .tools import get_stock_data


class CoordinatorAgent:
    """Orchestrates Research -> Risk -> Reporting in a deterministic sequence."""

    def __init__(self) -> None:
        self.research_agent = ResearchAgent()
        self.risk_agent = RiskAgent()
        self.reporting_agent = ReportingAgent()

    def run(
        self,
        ticker: str,
        period: str = "1y",
        use_mock_data: bool = False,
    ) -> dict[str, Any]:
        # Stage 1: data gathering / research
        prices = get_stock_data(ticker, period=period, use_mock_data=use_mock_data)
        research = self.research_agent.run(ticker, prices)

        # Stage 2: risk analysis consumes the same validated price series
        risk = self.risk_agent.run(prices)

        # Stage 3: reporting consumes both upstream outputs
        report = self.reporting_agent.run(ticker, research, risk)

        return {
            "ticker": ticker.upper(),
            "research": research,
            "risk": risk.to_dict(),
            "report": report,
        }
