"""
tests/test_decision_agent.py
----------------------------
Tests du Decision Agent.

Lance : uv run pytest tests/test_decision_agent.py -v
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from execution.decision_agent import DecisionAgent, TradeOrder
from signals.aggregator import SignalVector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_vector(
    symbol: str = "AAPL",
    momentum_composite: float | None = 0.8,
    value: float | None = 0.3,
    quality: float | None = 0.5,
    low_volatility: float | None = 0.2,
    data_quality: float = 1.0,
    n_bars: int = 500,
) -> SignalVector:
    return SignalVector(
        symbol=symbol,
        timestamp=datetime(2025, 1, 15, 12, 0),
        momentum_composite=momentum_composite,
        momentum_3m=0.6,
        momentum_12m=0.7,
        value=value,
        quality=quality,
        low_volatility=low_volatility,
        data_quality=data_quality,
        n_bars=n_bars,
    )


VALID_LONG_RESPONSE = """{
    "direction": "LONG",
    "confidence": 0.78,
    "entry": 0.0,
    "stop_loss": 0.02,
    "take_profit": 0.05,
    "size_pct": 0.04,
    "rationale": "Strong momentum composite with supportive quality score."
}"""

VALID_SHORT_RESPONSE = """{
    "direction": "SHORT",
    "confidence": 0.65,
    "entry": 0.0,
    "stop_loss": 0.015,
    "take_profit": 0.04,
    "size_pct": 0.02,
    "rationale": "Negative momentum signals across horizons."
}"""

LOW_CONFIDENCE_RESPONSE = """{
    "direction": "LONG",
    "confidence": 0.45,
    "entry": 0.0,
    "stop_loss": 0.02,
    "take_profit": 0.04,
    "size_pct": 0.03,
    "rationale": "Weak signal, uncertain direction."
}"""

FLAT_RESPONSE = """{
    "direction": "FLAT",
    "confidence": 0.55,
    "entry": 0.0,
    "stop_loss": 0.02,
    "take_profit": 0.04,
    "size_pct": 0.0,
    "rationale": "Conflicting signals, staying flat."
}"""


def make_agent_with_mock(response_text: str) -> DecisionAgent:
    """Crée un DecisionAgent dont l'appel Claude est mocké."""
    agent = DecisionAgent.__new__(DecisionAgent)
    agent.model = "claude-sonnet-4-6"
    agent.max_tokens = 512
    agent.temperature = 0.2

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = response_text
    mock_message.content = [mock_content]
    mock_client.messages.create.return_value = mock_message
    agent._client = mock_client

    return agent


# ---------------------------------------------------------------------------
# Tests TradeOrder
# ---------------------------------------------------------------------------


class TestTradeOrder:
    def test_flat_when_confidence_low(self):
        order = TradeOrder(
            symbol="AAPL",
            direction="LONG",
            confidence=0.4,
            size_pct=0.03,
        )
        assert order.direction == "FLAT"
        assert order.size_pct == 0.0

    def test_direction_preserved_when_confidence_high(self):
        order = TradeOrder(
            symbol="AAPL",
            direction="LONG",
            confidence=0.75,
            size_pct=0.03,
        )
        assert order.direction == "LONG"
        assert order.size_pct == 0.03

    def test_short_preserved_when_confidence_high(self):
        order = TradeOrder(
            symbol="SPY",
            direction="SHORT",
            confidence=0.65,
            size_pct=0.02,
        )
        assert order.direction == "SHORT"

    def test_is_active_true_for_long(self):
        order = TradeOrder(symbol="X", direction="LONG", confidence=0.8, size_pct=0.04)
        assert order.is_active() is True

    def test_is_active_false_for_flat(self):
        order = TradeOrder(symbol="X", direction="FLAT", confidence=0.9, size_pct=0.0)
        assert order.is_active() is False

    def test_to_dict_contains_all_fields(self):
        order = TradeOrder(symbol="AAPL", direction="LONG", confidence=0.8)
        d = order.to_dict()
        for key in (
            "symbol",
            "direction",
            "confidence",
            "entry",
            "stop_loss",
            "take_profit",
            "size_pct",
            "rationale",
        ):
            assert key in d

    def test_exact_confidence_060_is_active(self):
        """confidence == 0.6 est à la limite — doit rester LONG."""
        order = TradeOrder(symbol="X", direction="LONG", confidence=0.6, size_pct=0.01)
        assert order.direction == "LONG"

    def test_confidence_059_is_flat(self):
        order = TradeOrder(symbol="X", direction="LONG", confidence=0.59, size_pct=0.01)
        assert order.direction == "FLAT"
        assert order.size_pct == 0.0


# ---------------------------------------------------------------------------
# Tests DecisionAgent._parse_order
# ---------------------------------------------------------------------------


class TestParseOrder:
    def setup_method(self):
        self.agent = make_agent_with_mock("")

    def test_parse_valid_long(self):
        order = self.agent._parse_order(VALID_LONG_RESPONSE, "AAPL")
        assert order.direction == "LONG"
        assert order.confidence == pytest.approx(0.78)
        assert order.size_pct == pytest.approx(0.04)
        assert "momentum" in order.rationale.lower()

    def test_parse_valid_short(self):
        order = self.agent._parse_order(VALID_SHORT_RESPONSE, "SPY")
        assert order.direction == "SHORT"
        assert order.confidence == pytest.approx(0.65)

    def test_parse_low_confidence_forces_flat(self):
        order = self.agent._parse_order(LOW_CONFIDENCE_RESPONSE, "AAPL")
        assert order.direction == "FLAT"
        assert order.size_pct == 0.0

    def test_parse_invalid_json_returns_flat(self):
        order = self.agent._parse_order("not valid json !!!", "AAPL")
        assert order.direction == "FLAT"
        assert order.confidence == 0.0

    def test_parse_invalid_direction_defaults_flat(self):
        bad = '{"direction": "HOLD", "confidence": 0.8, "entry": 0.0, "stop_loss": 0.02, "take_profit": 0.04, "size_pct": 0.03, "rationale": "test"}'
        order = self.agent._parse_order(bad, "AAPL")
        assert order.direction == "FLAT"

    def test_parse_size_pct_capped_at_5pct(self):
        oversized = '{"direction": "LONG", "confidence": 0.9, "entry": 0.0, "stop_loss": 0.02, "take_profit": 0.04, "size_pct": 0.50, "rationale": "test"}'
        order = self.agent._parse_order(oversized, "AAPL")
        assert order.size_pct <= 0.05

    def test_parse_strips_markdown_codeblock(self):
        wrapped = f"```json\n{VALID_LONG_RESPONSE}\n```"
        order = self.agent._parse_order(wrapped, "AAPL")
        assert order.direction == "LONG"

    def test_parse_confidence_clamped_to_01(self):
        bad_conf = '{"direction": "LONG", "confidence": 1.5, "entry": 0.0, "stop_loss": 0.02, "take_profit": 0.04, "size_pct": 0.03, "rationale": "test"}'
        order = self.agent._parse_order(bad_conf, "AAPL")
        assert order.confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests DecisionAgent.decide (intégration mockée)
# ---------------------------------------------------------------------------


class TestDecideMethod:
    def test_decide_returns_trade_order(self):
        agent = make_agent_with_mock(VALID_LONG_RESPONSE)
        vector = make_vector()
        order = agent.decide(vector)
        assert isinstance(order, TradeOrder)
        assert order.symbol == "AAPL"

    def test_decide_long_signal(self):
        agent = make_agent_with_mock(VALID_LONG_RESPONSE)
        vector = make_vector(momentum_composite=0.9, quality=0.8)
        order = agent.decide(vector)
        assert order.direction == "LONG"
        assert order.is_active()

    def test_decide_low_confidence_is_flat(self):
        agent = make_agent_with_mock(LOW_CONFIDENCE_RESPONSE)
        vector = make_vector()
        order = agent.decide(vector)
        assert order.direction == "FLAT"
        assert not order.is_active()

    def test_decide_calls_claude_once(self):
        agent = make_agent_with_mock(VALID_LONG_RESPONSE)
        vector = make_vector()
        agent.decide(vector)
        agent._client.messages.create.assert_called_once()

    def test_decide_passes_signal_in_prompt(self):
        agent = make_agent_with_mock(FLAT_RESPONSE)
        vector = make_vector(symbol="TSLA")
        agent.decide(vector)
        call_kwargs = agent._client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "TSLA" in user_content

    def test_decide_short_signal(self):
        agent = make_agent_with_mock(VALID_SHORT_RESPONSE)
        vector = make_vector(momentum_composite=-0.7, value=-0.5)
        order = agent.decide(vector)
        assert order.direction == "SHORT"

    def test_decide_uses_correct_model(self):
        agent = make_agent_with_mock(VALID_LONG_RESPONSE)
        vector = make_vector()
        agent.decide(vector)
        call_kwargs = agent._client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Test intégration réelle (nécessite ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDecisionAgentIntegration:
    """Lance avec : uv run pytest -m integration"""

    def test_real_decide_aapl(self):
        import numpy as np
        import pandas as pd

        from signals.aggregator import SignalAggregator

        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns = rng.normal(0.0003, 0.015, n)
        close = 100 * np.exp(np.cumsum(returns))
        noise = rng.uniform(0.005, 0.015, n)
        prices = pd.DataFrame(
            {
                "open": close * (1 - noise / 2),
                "high": close * (1 + noise),
                "low": close * (1 - noise),
                "close": close,
                "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )

        agg = SignalAggregator(min_bars=100)
        vector = agg.compute(prices, symbol="AAPL")

        agent = DecisionAgent()
        order = agent.decide(vector)

        assert isinstance(order, TradeOrder)
        assert order.direction in ("LONG", "SHORT", "FLAT")
        assert 0.0 <= order.confidence <= 1.0
        assert 0.0 <= order.size_pct <= 0.05
        assert len(order.rationale) > 0
