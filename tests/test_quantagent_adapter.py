"""
tests/test_quantagent_adapter.py
---------------------------------
Tests unitaires du QuantAgentAdapter.

Aucun appel LLM réel — tout est mocké.

Lance : uv run pytest tests/test_quantagent_adapter.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from signals.agents.quantagent_adapter import AgentSignal, QuantAgentAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    """DataFrame OHLCV minimal (60 lignes) au format OpenBB."""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(60)],
            "high": [101.0 + i * 0.1 for i in range(60)],
            "low": [99.0 + i * 0.1 for i in range(60)],
            "close": [100.5 + i * 0.1 for i in range(60)],
            "volume": [1_000_000] * 60,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. is_available() — submodule absent
# ---------------------------------------------------------------------------


def test_is_available_returns_false_when_submodule_missing():
    adapter = QuantAgentAdapter()
    with patch(
        "signals.agents.quantagent_adapter._QUANTAGENT_PATH",
        Path("/nonexistent/path"),
    ):
        assert adapter.is_available() is False


# ---------------------------------------------------------------------------
# 2. analyze() — retourne AgentSignal avec agent_bias=None si LLM indispo
# ---------------------------------------------------------------------------


def test_analyze_returns_none_bias_when_unavailable(prices):
    adapter = QuantAgentAdapter()
    with patch.object(adapter, "is_available", return_value=False):
        signal = adapter.analyze(prices, symbol="AAPL")

    assert isinstance(signal, AgentSignal)
    assert signal.agent_bias is None
    assert signal.direction == "UNKNOWN"


# ---------------------------------------------------------------------------
# 3. _parse_decision() : mapping LONG / SHORT / FLAT
# ---------------------------------------------------------------------------


class TestParseDecision:
    def test_long_json(self):
        raw = '{"decision": "LONG", "forecast_horizon": "next 3 candles", "justification": "bullish", "risk_reward_ratio": 1.5}'
        direction, bias, confidence = QuantAgentAdapter._parse_decision(raw)
        assert direction == "LONG"
        assert bias is not None and bias > 0
        assert confidence is not None and 0.0 <= confidence <= 1.0

    def test_short_json(self):
        raw = '{"decision": "SHORT", "forecast_horizon": "next candle", "justification": "bearish", "risk_reward_ratio": 1.8}'
        direction, bias, confidence = QuantAgentAdapter._parse_decision(raw)
        assert direction == "SHORT"
        assert bias is not None and bias < 0

    def test_long_extreme_bias(self):
        raw = '{"decision": "LONG", "risk_reward_ratio": 1.8}'
        _, bias, _ = QuantAgentAdapter._parse_decision(raw)
        assert bias == pytest.approx(1.0, abs=0.01)

    def test_long_min_bias(self):
        raw = '{"decision": "LONG", "risk_reward_ratio": 1.2}'
        _, bias, _ = QuantAgentAdapter._parse_decision(raw)
        assert bias == pytest.approx(0.3, abs=0.01)

    def test_short_mirrors_long(self):
        long_raw = '{"decision": "LONG", "risk_reward_ratio": 1.5}'
        short_raw = '{"decision": "SHORT", "risk_reward_ratio": 1.5}'
        _, long_bias, _ = QuantAgentAdapter._parse_decision(long_raw)
        _, short_bias, _ = QuantAgentAdapter._parse_decision(short_raw)
        assert long_bias == pytest.approx(-short_bias, abs=0.001)  # type: ignore

    def test_flat(self):
        direction, bias, _ = QuantAgentAdapter._parse_decision('{"decision": "FLAT", "risk_reward_ratio": 1.4}')
        assert direction == "FLAT"
        assert bias == 0.0

    def test_fallback_plain_text_long(self):
        direction, bias, _ = QuantAgentAdapter._parse_decision(
            "After analysis, the recommendation is LONG based on momentum."
        )
        assert direction == "LONG"
        assert bias is not None and bias > 0

    def test_fallback_plain_text_short(self):
        direction, bias, _ = QuantAgentAdapter._parse_decision("Market shows weakness — SHORT entry recommended.")
        assert direction == "SHORT"
        assert bias is not None and bias < 0

    def test_empty_string(self):
        direction, bias, confidence = QuantAgentAdapter._parse_decision("")
        assert direction == "UNKNOWN"
        assert bias is None
        assert confidence is None

    def test_json_with_backtick_fences(self):
        raw = '```json\n{"decision": "LONG", "risk_reward_ratio": 1.6}\n```'
        direction, bias, _ = QuantAgentAdapter._parse_decision(raw)
        assert direction == "LONG"
        assert bias is not None and bias > 0


# ---------------------------------------------------------------------------
# 4. analyze() ne lève jamais d'exception même si le LLM raise
# ---------------------------------------------------------------------------


def test_analyze_never_raises_on_llm_error(prices):
    adapter = QuantAgentAdapter()
    with (
        patch.object(adapter, "is_available", return_value=True),
        patch.object(adapter, "_run_analysis", side_effect=RuntimeError("LLM crash")),
    ):
        signal = adapter.analyze(prices, symbol="AAPL")

    assert isinstance(signal, AgentSignal)
    assert signal.agent_bias is None
    assert signal.direction == "UNKNOWN"


# ---------------------------------------------------------------------------
# Helper pour patcher l'import lazy dans aggregator.compute()
# L'import `from .agents.quantagent_adapter import QuantAgentAdapter` est lazy.
# Le patch doit remplacer la classe dans le module déjà en cache (sys.modules).
# ---------------------------------------------------------------------------

import signals.agents.quantagent_adapter as _qa_mod  # force le cache module

# ---------------------------------------------------------------------------
# 5. SignalAggregator.compute(use_quantagent=False) n'appelle pas QuantAgent
# ---------------------------------------------------------------------------


def test_aggregator_does_not_call_quantagent_by_default(prices):
    """Sans use_quantagent=True, agent_bias reste None — aucun appel LLM."""
    from signals.aggregator import SignalAggregator

    agg = SignalAggregator()
    vector = agg.compute(prices, symbol="TEST")  # use_quantagent=False par défaut
    assert vector.agent_bias is None


# ---------------------------------------------------------------------------
# 6. SignalAggregator.compute(use_quantagent=True) appelle l'adaptateur
# ---------------------------------------------------------------------------


def test_aggregator_calls_quantagent_when_requested(prices):
    from signals.aggregator import SignalAggregator

    mock_adapter = MagicMock()
    mock_adapter.is_available.return_value = True
    mock_adapter.analyze.return_value = AgentSignal(
        agent_bias=0.75,
        direction="LONG",
        latency_ms=123.0,
    )
    mock_cls = MagicMock(return_value=mock_adapter)

    original = _qa_mod.QuantAgentAdapter
    _qa_mod.QuantAgentAdapter = mock_cls  # type: ignore
    try:
        agg = SignalAggregator()
        vector = agg.compute(prices, symbol="AAPL", use_quantagent=True)
    finally:
        _qa_mod.QuantAgentAdapter = original

    mock_adapter.analyze.assert_called_once()
    assert vector.agent_bias == 0.75


# ---------------------------------------------------------------------------
# 7. agent_bias manuel a la priorité sur use_quantagent
# ---------------------------------------------------------------------------


def test_manual_agent_bias_takes_priority(prices):
    """Si agent_bias est passé manuellement, QuantAgent ne doit pas être appelé."""
    from signals.aggregator import SignalAggregator

    mock_adapter = MagicMock()
    mock_adapter.is_available.return_value = True
    mock_cls = MagicMock(return_value=mock_adapter)

    original = _qa_mod.QuantAgentAdapter
    _qa_mod.QuantAgentAdapter = mock_cls  # type: ignore
    try:
        agg = SignalAggregator()
        vector = agg.compute(prices, symbol="AAPL", agent_bias=0.42, use_quantagent=True)
    finally:
        _qa_mod.QuantAgentAdapter = original

    mock_adapter.analyze.assert_not_called()
    assert vector.agent_bias == 0.42


# ---------------------------------------------------------------------------
# 8. _to_kline_dict : format correct
# ---------------------------------------------------------------------------


def test_to_kline_dict_format(prices):
    adapter = QuantAgentAdapter()
    kline = adapter._to_kline_dict(prices)

    assert "Open" in kline
    assert "High" in kline
    assert "Low" in kline
    assert "Close" in kline
    assert len(kline["Open"]) == 30  # tail(30)
