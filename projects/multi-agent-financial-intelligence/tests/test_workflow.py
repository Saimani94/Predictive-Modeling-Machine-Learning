from financial_intelligence.coordinator import CoordinatorAgent


def test_sequential_workflow():
    result = CoordinatorAgent().run("AAPL", use_mock_data=True)

    assert result["ticker"] == "AAPL"
    assert result["research"]["observations"] == 252
    assert "Financial Intelligence Report" in result["report"]
    assert result["risk"]["risk_level"] in {"Low", "Moderate", "High"}
