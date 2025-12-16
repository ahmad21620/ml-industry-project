from unittest.mock import patch
import pytest

from tools.currency_fx_tool import CurrencyFXTool, FXAPIError


class DummyResponse:
    def __init__(self, status_code: int, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload


def test_exchangerate_host_uses_base_and_symbols_params():
    tool = CurrencyFXTool(
        base_url="https://api.exchangerate.host",
        api_key="",
        max_retries=0,
        timeout_seconds=1.0,
        enabled=True,
    )

    with patch("tools.currency_fx_tool.requests.get") as mock_get:
        mock_get.return_value = DummyResponse(
            200,
            payload={"base": "USD", "rates": {"EUR": 0.92}},
        )

        res = tool.get_rate("usd", "eur")

        assert res.base_currency == "USD"
        assert res.target_currency == "EUR"
        assert abs(res.rate - 0.92) < 1e-12
        assert res.source == "ExchangeRate.host"

        # Verify request shape
        args, kwargs = mock_get.call_args
        assert args[0].endswith("/latest")
        assert kwargs["params"]["base"] == "USD"
        assert kwargs["params"]["symbols"] == "EUR"


def test_fallback_to_frankfurter_when_primary_fails():
    tool = CurrencyFXTool(
        base_url="https://api.exchangerate.host",
        api_key="",
        fallback_base_url="https://api.frankfurter.app",
        fallback_api_key="",
        enable_fallback=True,
        max_retries=0,
        timeout_seconds=1.0,
        enabled=True,
    )

    with patch("tools.currency_fx_tool.requests.get") as mock_get:
        # Primary: server error -> should trigger fallback
        # Fallback: success
        mock_get.side_effect = [
            DummyResponse(500, payload={"error": "server down"}, text="server down"),
            DummyResponse(
                200,
                payload={"amount": 1.0, "base": "USD", "rates": {"EUR": 0.93}},
            ),
        ]

        res = tool.get_rate("USD", "EUR")

        assert abs(res.rate - 0.93) < 1e-12
        assert res.source == "Frankfurter"
        assert mock_get.call_count == 2


def test_explicit_base_url_disables_fallback_by_default():
    tool = CurrencyFXTool(
        base_url="https://api.exchangerate.host",
        api_key="",
        fallback_base_url="https://api.frankfurter.app",
        fallback_api_key="",
        # enable_fallback not passed -> should default OFF because base_url is explicit
        max_retries=0,
        timeout_seconds=1.0,
        enabled=True,
    )

    with patch("tools.currency_fx_tool.requests.get") as mock_get:
        mock_get.return_value = DummyResponse(500, payload={"error": "server down"}, text="server down")

        with pytest.raises(FXAPIError) as exc:
            tool.get_rate("USD", "EUR")

        assert "fallback is disabled" in str(exc.value).lower()
        assert mock_get.call_count == 1