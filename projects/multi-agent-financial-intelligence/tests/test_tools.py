import numpy as np
import pandas as pd

from financial_intelligence.tools import assess_risk_metrics


def test_risk_metrics_are_finite():
    rng = np.random.default_rng(7)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 120))))
    result = assess_risk_metrics(prices)

    assert np.isfinite(result.annualized_volatility)
    assert np.isfinite(result.max_drawdown)
    assert np.isfinite(result.sharpe_ratio)
    assert np.isfinite(result.var_95)
    assert result.risk_level in {"Low", "Moderate", "High"}
