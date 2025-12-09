from typing import Union
from decimal import Decimal, ROUND_HALF_UP


Number = Union[int, float]


class CurrencyCalculatorTool:
    """Simple deterministic calculator for currency conversions.

    This tool performs only pure numerical operations and does not
    perform any network I/O.
    """

    def convert_amount(self, amount: Number, rate: Number, decimals: int = 2) -> float:
        """Convert `amount` using `rate` (base -> target).

        Uses decimal arithmetic with HALF_UP rounding (half away from zero)
        and returns a float suitable for display or further processing.
        """
        try:
            amount_dec = Decimal(str(amount))
            rate_dec = Decimal(str(rate))
        except Exception as exc:
            raise ValueError("Amount and rate must be numeric values.") from exc

        if rate_dec <= 0:
            raise ValueError(f"Rate must be a positive number, got {rate!r}.")

        if decimals < 0:
            raise ValueError("decimals must be a non-negative integer.")

        result = amount_dec * rate_dec

        # Build a quantization factor like 0.01 for decimals=2, 0.0001 for decimals=4, etc.
        quant = Decimal("1").scaleb(-decimals)

        rounded = result.quantize(quant, rounding=ROUND_HALF_UP)
        return float(rounded)
