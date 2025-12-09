from datetime import datetime

from agents.response_agent import ResponseAgent, Answer
from tools.currency_fx_tool import FXRateResult


class StubFXTool:
    """Stub FX tool that returns a fixed rate without real HTTP calls."""

    def __init__(self, rate: float = 2.0):
        self.rate = rate

    def get_rate(self, base_currency: str, target_currency: str) -> FXRateResult:
        return FXRateResult(
            base_currency=base_currency.upper(),
            target_currency=target_currency.upper(),
            rate=self.rate,
            fetched_at=datetime(2024, 1, 1, 0, 0, 0),
            source="stub",
        )


class StubCalculatorTool:
    def convert_amount(self, amount, rate, decimals: int = 2) -> float:
        # Simple multiplication with standard rounding
        return round(float(amount) * float(rate), decimals)


def test_direct_currency_conversion_answer():
    # rag_agent may be None here because direct conversion queries short-circuit
    stub_fx = StubFXTool(rate=2.0)
    stub_calc = StubCalculatorTool()

    agent = ResponseAgent(
        rag_agent=None,  # type: ignore[arg-type]
        kg_agent=None,
        currency_fx_tool=stub_fx,
        currency_calculator_tool=stub_calc,
    )

    question = "Convert 100 USD to EUR"
    answer: Answer = agent.answer(question)

    assert "100.00 USD" in answer.answer_text
    assert "200.00 EUR" in answer.answer_text
    assert "Using a live exchange rate" in answer.answer_text


def test_enhanced_aws_answer_with_conversion():
    stub_fx = StubFXTool(rate=2.0)
    stub_calc = StubCalculatorTool()

    agent = ResponseAgent(
        rag_agent=None,  # type: ignore[arg-type]
        kg_agent=None,
        currency_fx_tool=stub_fx,
        currency_calculator_tool=stub_calc,
    )

    # Simulate behavior of _maybe_enhance_answer_with_currency_conversion directly
    question = "Show my AWS bill for last month in EUR"
    raw_answer = "Your AWS bill for last month is 150.00 USD."

    enhanced = agent._maybe_enhance_answer_with_currency_conversion(
        question=question,
        answer_text=raw_answer,
    )

    assert "150.00 USD" in enhanced
    assert "300.00 EUR" in enhanced
    assert "Currency conversion (based on live exchange rates):" in enhanced
