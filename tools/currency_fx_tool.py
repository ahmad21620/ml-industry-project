from datetime import datetime, timezone
from typing import Optional

import requests
from pydantic import BaseModel

from config import (
    FX_PRIMARY_BASE_URL,
    FX_PRIMARY_API_KEY,
    FX_FALLBACK_BASE_URL,
    FX_FALLBACK_API_KEY,
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
        fallback_base_url: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        enable_fallback: Optional[bool] = None,
    ) -> None:
        # Primary provider (ExchangeRate.host by default)
        self.base_url = (base_url or FX_PRIMARY_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else FX_PRIMARY_API_KEY

        # Fallback provider (Frankfurter by default)
        self.fallback_base_url = (fallback_base_url or FX_FALLBACK_BASE_URL).rstrip("/")
        self.fallback_api_key = (
            fallback_api_key if fallback_api_key is not None else FX_FALLBACK_API_KEY
        )

        # If caller explicitly passes base_url, keep behavior deterministic:
        # fallback is OFF unless explicitly enabled.
        self.enable_fallback = (
            enable_fallback if enable_fallback is not None else (base_url is None)
        )

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

    @staticmethod
    def _provider_name(base_url: str) -> str:
        u = (base_url or "").lower()
        if "exchangerate.host" in u:
            return "ExchangeRate.host"
        if "frankfurter" in u:
            return "Frankfurter"
        return "FX_API"

    def _get_rate_from_provider(
        self,
        *,
        provider_base_url: str,
        provider_api_key: str,
        base: str,
        target: str,
    ) -> FXRateResult:
        """Fetch a rate from exactly one provider (no fallback here)."""
        logger.info(
            "FX request: %s -> %s (provider=%s)",
            base,
            target,
            provider_base_url,
        )

        # Trivial case: same currency.
        if base == target:
            fetched_at = datetime.now(timezone.utc)
            return FXRateResult(
                base_currency=base,
                target_currency=target,
                rate=1.0,
                fetched_at=fetched_at,
                source=self._provider_name(provider_base_url),
            )

        provider_l = provider_base_url.lower().rstrip("/")

        # APILayer ExchangeRate.host endpoints (require access_key):
        # https://api.exchangerate.host/convert?from=EUR&to=GBP&amount=100&access_key=KEY
        # https://api.exchangerate.host/live?access_key=KEY
        # https://api.exchangerate.host/list?access_key=KEY
        if "exchangerate.host" in provider_l:
            url = f"{provider_base_url.rstrip('/')}/convert"
            params = {"from": base, "to": target, "amount": 1}
            if provider_api_key:
                params["access_key"] = provider_api_key
        else:
            # Frankfurter-like providers
            url = f"{provider_base_url.rstrip('/')}/latest"
            params = {"from": base, "to": target, "amount": 1}
            if provider_api_key:
                params["apikey"] = provider_api_key


        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "FX HTTP attempt %s for %s -> %s (provider=%s) params=%r",
                    attempt + 1,
                    base,
                    target,
                    provider_base_url,
                    params,
                )
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.RequestException,
            ) as exc:
                last_error = exc
                logger.warning(
                    "FX HTTP error on attempt %s for %s -> %s (provider=%s): %r",
                    attempt + 1,
                    base,
                    target,
                    provider_base_url,
                    exc,
                )
                continue

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

            # Some providers include: {"success": false, "error": {...}}
            if isinstance(data, dict) and data.get("success") is False:
                err_msg = None
                if isinstance(data.get("error"), dict):
                    err_msg = data["error"].get("info") or data["error"].get("type")
                raise FXAPIError(f"FX API indicated failure: {err_msg or data!r}")

            try:
                rate_value = None

                if isinstance(data, dict):
                    # APILayer convert endpoint usually returns "result" (converted amount)
                    # and/or info.quote (rate). We call amount=1, so result == rate.
                    if "result" in data and data["result"] is not None:
                        rate_value = float(data["result"])
                    else:
                        info = data.get("info")
                        if isinstance(info, dict):
                            if "quote" in info and info["quote"] is not None:
                                rate_value = float(info["quote"])
                            elif "rate" in info and info["rate"] is not None:
                                rate_value = float(info["rate"])

                    # Frankfurter-like fallback
                    if rate_value is None:
                        rates = data.get("rates")
                        if isinstance(rates, dict) and target in rates:
                            rate_value = float(rates[target])

                if rate_value is None:
                    raise KeyError("Rate not found in response.")
            except (KeyError, TypeError, ValueError) as exc:
                raise FXAPIError("FX API response was missing expected rate data.") from exc


            if rate_value <= 0:
                raise FXAPIError(f"FX API returned a non-positive rate: {rate_value!r}")

            fetched_at = datetime.now(timezone.utc)
            source_name = self._provider_name(provider_base_url)

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

        raise FXAPIError(
            f"Failed to fetch FX rate after {self.max_retries + 1} attempts."
        ) from last_error


    def get_rate(self, base_currency: str, target_currency: str) -> FXRateResult:
        """Return the FX rate from base_currency to target_currency.

        Primary: ExchangeRate.host (default)
        Fallback: Frankfurter (default)
        """
        if not self.enabled:
            raise FXAPIError("Currency FX API is disabled by configuration.")

        base = self._normalize_currency_code(base_currency)
        target = self._normalize_currency_code(target_currency)

        primary_err: Optional[Exception] = None

        # 1) Try primary
        try:
            return self._get_rate_from_provider(
                provider_base_url=self.base_url,
                provider_api_key=self.api_key or "",
                base=base,
                target=target,
            )
        except FXAPIError as exc:
            primary_err = exc

        # 2) Try fallback (only when enabled and configured)
        if (
            self.enable_fallback
            and self.fallback_base_url
            and self.fallback_base_url.rstrip("/") != self.base_url.rstrip("/")
        ):
            logger.warning(
                "FX primary failed (%s). Falling back to %s",
                str(primary_err) or repr(primary_err),
                self.fallback_base_url,
            )
            try:
                return self._get_rate_from_provider(
                    provider_base_url=self.fallback_base_url,
                    provider_api_key=self.fallback_api_key or "",
                    base=base,
                    target=target,
                )
            except FXAPIError as fallback_exc:
                raise FXAPIError(
                    "FX rate fetch failed using both providers. "
                    f"primary_error={primary_err!s}; fallback_error={fallback_exc!s}"
                ) from fallback_exc

        # No fallback allowed/configured
        raise FXAPIError(
            "FX rate fetch failed using primary provider and fallback is disabled. "
            f"primary_error={primary_err!s}"
        ) from primary_err
