from datetime import datetime

import pytest

from tools.currency_fx_tool import CurrencyFXTool, FXRateResult, FXAPIError


class DummyResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    @property
    def text(self):
        return str(self._json_data)


def test_get_rate_success(monkeypatch):
    """Ensure that CurrencyFXTool parses a basic successful response."""

    def dummy_get(url, params=None, timeout=None):
        assert params["from"] == "USD"
        assert params["to"] == "EUR"
        return DummyResponse(
            200,
            {
                "amount": 1.0,
                "base": "USD",
                "date": "2024-01-01",
                "rates": {"EUR": 0.9},
            },
        )

    monkeypatch.setattr("tools.currency_fx_tool.requests.get", dummy_get)

    tool = CurrencyFXTool(base_url="https://api.example.com")

    result = tool.get_rate("usd", "eur")
    assert isinstance(result, FXRateResult)
    assert result.base_currency == "USD"
    assert result.target_currency == "EUR"
    assert result.rate == 0.9


def test_get_rate_invalid_code():
    tool = CurrencyFXTool()

    with pytest.raises(FXAPIError):
        tool.get_rate("US", "EUR")  # invalid base code

    with pytest.raises(FXAPIError):
        tool.get_rate("USD", "EU")  # invalid target code


def test_get_rate_same_currency():
    tool = CurrencyFXTool()
    result = tool.get_rate("usd", "usd")
    assert result.rate == 1.0
    assert result.base_currency == "USD"
    assert result.target_currency == "USD"
    assert isinstance(result.fetched_at, datetime)
