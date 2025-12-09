from tools.calculator_tool import CurrencyCalculatorTool


def test_convert_amount_basic():
    calc = CurrencyCalculatorTool()
    result = calc.convert_amount(100, 1.5)
    # 100 * 1.5 = 150.0
    assert result == 150.0


def test_convert_amount_rounding():
    calc = CurrencyCalculatorTool()
    # 1 / 3 * 10 = 3.333..., rounded to 2 decimals -> 3.33
    result = calc.convert_amount(10, 1 / 3, decimals=2)
    assert round(result, 2) == 3.33


def test_convert_amount_rejects_non_positive_rate():
    calc = CurrencyCalculatorTool()
    try:
        calc.convert_amount(100, 0)
        assert False, "Expected ValueError for zero rate"
    except ValueError:
        pass