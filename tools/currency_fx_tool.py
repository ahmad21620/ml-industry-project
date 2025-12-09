from datetime import datetime
from typing import Optional

import requests
from pydantic import BaseModel

from config import (
    FX_API_BASE_URL,
    FX_API_KEY,
    FX_API_TIMEOUT_SECONDS,
    FX_API_MAX_RETRIES,
    FX_API_ENABLED,
)

import logging


logger = logging.getLogger(__name__)


class FXAPIError(Exception):
    """Domain-specific error for FX API issues."""


class FXRateResult(BaseModel):
    base_currency: str
    target_currency: str
    rate: float
    fetched_at: datetime
    source: str


class CurrencyFXTool:
    """Tool responsible for fetching live FX rates from an external API.

    This implementation uses a configurable HTTP endpoint (default:
    Frankfurter API) and never accepts exchange rates from the caller.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.base_url = (base_url or FX_API_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else FX_API_KEY
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else FX_API_TIMEOUT_SECONDS
        )
        self.max_retries = (
            max_retries if max_retries is not None else FX_API_MAX_RETRIES
        )
        self.enabled = FX_API_ENABLED if enabled is None else enabled

    @staticmethod
    def _normalize_currency_code(code: str) -> str:
        """Normalize and validate a currency code as ISO 4217 (3 letters)."""
        if code is None:
            raise FXAPIError("Currency code must not be None.")
        normalized = code.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise FXAPIError(f"Invalid currency code: {code!r}")
        return normalized

    def get_rate(self, base_currency: str, target_currency: str) -> FXRateResult:
        """Return the FX rate from base_currency to target_currency.

        The rate is always obtained from the external FX API; no user-supplied
        rate is ever trusted or used here.
        """
        if not self.enabled:
            raise FXAPIError("Currency FX API is disabled by configuration.")

        base = self._normalize_currency_code(base_currency)
        target = self._normalize_currency_code(target_currency)

        logger.info(
            "FX request: %s -> %s (base_url=%s, enabled=%s)",
            base,
            target,
            self.base_url,
            self.enabled,
        )

        # Trivial case: same currency.
        if base == target:
            fetched_at = datetime.utcnow()
            source_name = (
                "Frankfurter" if "frankfurter" in self.base_url.lower() else "FX_API"
            )
            logger.info(
                "FX request %s -> %s: trivial same-currency rate=1.0",
                base,
                target,
            )
            return FXRateResult(
                base_currency=base,
                target_currency=target,
                rate=1.0,
                fetched_at=fetched_at,
                source=source_name,
            )

        url = f"{self.base_url}/latest"
        params = {
            "from": base,
            "to": target,
            "amount": 1,
        }

        # Generic API key support; actual provider may ignore or use this.
        if self.api_key:
            # Many FX providers accept an "apikey" query parameter; if you
            # switch providers, adjust this to match their requirements.
            params["apikey"] = self.api_key

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "FX HTTP attempt %s for %s -> %s with params=%r",
                    attempt + 1,
                    base,
                    target,
                    params,
                )
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds,
                )
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.RequestException,
            ) as exc:
                logger.warning(
                    "FX HTTP error on attempt %s for %s -> %s: %r",
                    attempt + 1,
                    base,
                    target,
                    exc,
                )
                last_error = exc
                continue

            # Retry on server errors, fail fast on others.
            if response.status_code >= 500:
                last_error = FXAPIError(
                    f"FX API server error (status {response.status_code})."
                )
                continue

            if response.status_code != 200:
                raise FXAPIError(
                    f"FX API returned status {response.status_code}: {response.text}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise FXAPIError("FX API returned invalid JSON.") from exc

            try:
                # Frankfurter-like schema:
                # {
                #   "amount": 1.0,
                #   "base": "USD",
                #   "date": "2024-01-01",
                #   "rates": { "EUR": 0.92 }
                # }
                rates = data["rates"]
                rate_value = float(rates[target])
            except (KeyError, TypeError, ValueError) as exc:
                raise FXAPIError(
                    "FX API response was missing expected rate data."
                ) from exc

            if rate_value <= 0:
                raise FXAPIError(
                    f"FX API returned a non-positive rate: {rate_value!r}"
                )

            fetched_at = datetime.utcnow()
            source_name = (
                "Frankfurter" if "frankfurter" in self.base_url.lower() else "FX_API"
            )

            logger.info(
                "FX request %s -> %s succeeded with rate=%f (source=%s)",
                base,
                target,
                rate_value,
                source_name,
            )

            return FXRateResult(
                base_currency=base,
                target_currency=target,
                rate=rate_value,
                fetched_at=fetched_at,
                source=source_name,
            )

        # If we reach here, all attempts failed.
        logger.error(
            "FX request %s -> %s failed after %s attempts. last_error=%r",
            base,
            target,
            self.max_retries + 1,
            last_error,
        )
        raise FXAPIError(
            f"Failed to fetch FX rate after {self.max_retries + 1} attempts."
        ) from last_error