"""Command-line entry point."""

import argparse

from financial_intelligence.coordinator import CoordinatorAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Financial Intelligence System")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period", default="1y", help="yfinance period, e.g. 6mo, 1y, 2y")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock data")
    args = parser.parse_args()

    result = CoordinatorAgent().run(
        ticker=args.ticker,
        period=args.period,
        use_mock_data=args.mock,
    )
    print(result["report"])


if __name__ == "__main__":
    main()
