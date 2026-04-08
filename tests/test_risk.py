"""
tests/test_risk.py
------------------
Tests unitaires du RiskEngine et Portfolio.
Aucun appel LLM ni OpenBB — données synthétiques uniquement.

Lance : uv run pytest tests/test_risk.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from execution.decision_agent import DecisionAgent, TradeOrder
from execution.risk import Portfolio, RiskEngine, RiskLimits
from signals.aggregator import SignalVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_order(
    symbol: str = "AAPL",
    direction: str = "LONG",
    confidence: float = 0.75,
    size_pct: float = 0.03,
) -> TradeOrder:
    return TradeOrder(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        size_pct=size_pct,
    )


def make_engine(
    cash: float = 100_000.0,
    positions: dict | None = None,
    equity_history: list[float] | None = None,
    limits: RiskLimits | None = None,
) -> RiskEngine:
    portfolio = Portfolio(
        cash=cash,
        positions=positions or {},
        equity_history=equity_history or [],
    )
    return RiskEngine(portfolio, limits or RiskLimits())


def normal_returns(n: int = 252, daily_vol: float = 0.01) -> list[float]:
    """Rendements synthétiques faibles (VaR < 2%)."""
    import random
    rng = random.Random(42)
    return [rng.gauss(0.0003, daily_vol) for _ in range(n)]


def fat_tail_returns(n: int = 252) -> list[float]:
    """Rendements avec queue grasse pour déclencher la VaR."""
    import random
    rng = random.Random(99)
    returns = [rng.gauss(0.0, 0.03) for _ in range(n)]
    # Force some extreme losses to push VaR above 2%
    returns[:10] = [-0.05] * 10
    return returns


# ---------------------------------------------------------------------------
# Tests Portfolio
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_total_equity_cash_only(self):
        p = Portfolio(cash=100_000)
        assert p.total_equity == pytest.approx(100_000)

    def test_total_equity_with_positions(self):
        p = Portfolio(cash=80_000, positions={"AAPL": 15_000, "TSLA": -5_000})
        assert p.total_equity == pytest.approx(100_000)

    def test_position_pct_known_symbol(self):
        p = Portfolio(cash=90_000, positions={"AAPL": 10_000})
        assert p.position_pct("AAPL") == pytest.approx(0.10)

    def test_position_pct_missing_symbol(self):
        p = Portfolio(cash=100_000)
        assert p.position_pct("TSLA") == pytest.approx(0.0)

    def test_portfolio_max_drawdown(self):
        """max_drawdown() calcul correct."""
        equity = [100_000, 105_000, 110_000, 95_000, 88_000, 92_000]
        p = Portfolio(equity_history=equity)
        dd = p.max_drawdown()
        # Peak = 110_000, trough = 88_000 → (88000-110000)/110000 ≈ -0.20
        assert dd == pytest.approx((88_000 - 110_000) / 110_000, abs=1e-6)
        assert dd < 0

    def test_portfolio_max_drawdown_no_history(self):
        p = Portfolio(equity_history=[])
        assert p.max_drawdown() == pytest.approx(0.0)

    def test_portfolio_max_drawdown_monotone_growth(self):
        p = Portfolio(equity_history=[100, 110, 120, 130])
        assert p.max_drawdown() == pytest.approx(0.0)

    def test_portfolio_var_95(self):
        """var_95() calcul correct avec returns synthétiques."""
        returns = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 95
        p = Portfolio()
        var = p.var_95(returns)
        # 5th percentile of 100 values → index 4 → -0.01, VaR = 0.01 (positive)
        assert var >= 0.0
        # The worst 5% of losses should be captured
        assert var >= 0.01

    def test_portfolio_var_95_empty(self):
        p = Portfolio()
        assert p.var_95([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests RiskEngine — position size
# ---------------------------------------------------------------------------


class TestPositionSize:
    def test_position_size_reduced(self):
        """size_pct 8% → réduit à 5% sans bloquer."""
        engine = make_engine()
        order = make_order(size_pct=0.08)
        result = engine.validate(order)
        assert result is not None
        assert result.size_pct == pytest.approx(0.05)

    def test_position_size_within_limit_unchanged(self):
        engine = make_engine()
        order = make_order(size_pct=0.03)
        result = engine.validate(order)
        assert result is not None
        assert result.size_pct == pytest.approx(0.03)

    def test_position_size_exactly_at_limit_unchanged(self):
        engine = make_engine()
        order = make_order(size_pct=0.05)
        result = engine.validate(order)
        assert result is not None
        assert result.size_pct == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Tests RiskEngine — drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_order_blocked_drawdown(self):
        """Portfolio en drawdown -12% → order bloqué."""
        # Peak 100k, now at 88k → drawdown = -12%
        equity_history = [100_000, 88_000]
        engine = make_engine(cash=88_000, equity_history=equity_history)
        order = make_order()
        result = engine.validate(order)
        assert result is None

    def test_order_passes_acceptable_drawdown(self):
        """Drawdown de -5% (< limite 10%) → order passe."""
        equity_history = [100_000, 95_000]
        engine = make_engine(cash=95_000, equity_history=equity_history)
        order = make_order()
        result = engine.validate(order)
        assert result is not None

    def test_order_blocked_drawdown_exactly_at_limit(self):
        """Drawdown exact = -10% → bloqué (strict >)."""
        equity_history = [100_000, 90_000]
        engine = make_engine(cash=90_000, equity_history=equity_history)
        order = make_order()
        result = engine.validate(order)
        assert result is None


# ---------------------------------------------------------------------------
# Tests RiskEngine — VaR
# ---------------------------------------------------------------------------


class TestVaR:
    def test_var_too_high_blocks_order(self):
        """VaR estimée > limite → order bloqué."""
        engine = make_engine()
        order = make_order()
        bad_returns = fat_tail_returns()
        result = engine.validate(order, recent_returns=bad_returns)
        assert result is None

    def test_var_acceptable_passes(self):
        """VaR < 2% → order passe."""
        engine = make_engine()
        order = make_order()
        good_returns = normal_returns(252, daily_vol=0.005)
        result = engine.validate(order, recent_returns=good_returns)
        assert result is not None

    def test_var_check_skipped_when_few_returns(self):
        """Moins de 20 données → VaR ignorée."""
        engine = make_engine()
        order = make_order()
        # Only 5 returns — check skipped, order should pass
        result = engine.validate(order, recent_returns=[-0.10] * 5)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests RiskEngine — confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_low_confidence_blocked(self):
        engine = make_engine()
        order = TradeOrder(symbol="AAPL", direction="LONG", confidence=0.4, size_pct=0.03)
        # __post_init__ forces FLAT for confidence < 0.6, so test with engine directly
        # Manually set direction to bypass __post_init__ effect
        order.direction = "LONG"
        order.confidence = 0.4
        result = engine.validate(order)
        assert result is None

    def test_confidence_at_limit_passes(self):
        engine = make_engine()
        order = make_order(confidence=0.60)
        result = engine.validate(order)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests RiskEngine — FLAT orders
# ---------------------------------------------------------------------------


class TestFlatOrders:
    def test_flat_order_always_passes(self):
        """Direction FLAT → toujours validé, même en drawdown."""
        equity_history = [100_000, 80_000]  # -20% drawdown
        engine = make_engine(cash=80_000, equity_history=equity_history)
        order = TradeOrder(symbol="AAPL", direction="FLAT", confidence=0.9, size_pct=0.0)
        result = engine.validate(order)
        assert result is not None
        assert result.direction == "FLAT"

    def test_flat_order_passes_with_high_var(self):
        """FLAT passe même avec VaR élevée."""
        engine = make_engine()
        order = TradeOrder(symbol="AAPL", direction="FLAT", confidence=0.9, size_pct=0.0)
        result = engine.validate(order, recent_returns=fat_tail_returns())
        assert result is not None


# ---------------------------------------------------------------------------
# Tests RiskEngine — conditions normales
# ---------------------------------------------------------------------------


class TestNormalConditions:
    def test_order_passes_normal_conditions(self):
        """Conditions normales → ordre passe inchangé."""
        engine = make_engine(
            cash=100_000,
            equity_history=[95_000, 98_000, 100_000],  # légère hausse
        )
        order = make_order(size_pct=0.03)
        result = engine.validate(order, recent_returns=normal_returns())
        assert result is not None
        assert result.direction == "LONG"
        assert result.size_pct == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Tests audit_log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_audit_log_records_all_decisions(self):
        """Chaque validate() laisse une trace dans audit_log."""
        engine = make_engine()

        order1 = make_order(symbol="AAPL", size_pct=0.03)
        order2 = make_order(symbol="TSLA", size_pct=0.08)  # will be reduced
        order3 = TradeOrder(symbol="SPY", direction="FLAT", confidence=0.9)

        engine.validate(order1)
        engine.validate(order2)
        engine.validate(order3)

        log = engine.audit_log()
        assert len(log) >= 3

    def test_audit_log_contains_required_fields(self):
        engine = make_engine()
        order = make_order()
        engine.validate(order)

        log = engine.audit_log()
        assert len(log) >= 1
        entry = log[0]
        for key in ("timestamp", "symbol", "outcome", "rule", "observed", "limit"):
            assert key in entry

    def test_audit_log_is_copy(self):
        """audit_log() retourne une copie — modifications externes sans effet."""
        engine = make_engine()
        engine.validate(make_order())
        log = engine.audit_log()
        original_len = len(log)
        log.append({"fake": True})
        assert len(engine.audit_log()) == original_len

    def test_audit_log_blocked_recorded(self):
        """Un ordre bloqué apparaît dans le log avec outcome=blocked."""
        equity_history = [100_000, 85_000]
        engine = make_engine(cash=85_000, equity_history=equity_history)
        engine.validate(make_order())

        log = engine.audit_log()
        blocked = [e for e in log if e["outcome"] == "blocked"]
        assert len(blocked) >= 1
        assert blocked[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Test intégration decide_with_risk
# ---------------------------------------------------------------------------


def make_decision_agent_mock(response_text: str) -> DecisionAgent:
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


LONG_RESPONSE = """{
    "direction": "LONG",
    "confidence": 0.78,
    "entry": 0.0,
    "stop_loss": 0.02,
    "take_profit": 0.05,
    "size_pct": 0.04,
    "rationale": "Strong momentum."
}"""


class TestDecideWithRisk:
    def test_decide_with_risk_integration(self):
        """decide_with_risk() retourne None si risk engine bloque (drawdown)."""
        agent = make_decision_agent_mock(LONG_RESPONSE)
        vector = SignalVector(symbol="AAPL", n_bars=500, data_quality=1.0)

        # Portfolio en drawdown -15%
        portfolio = Portfolio(
            cash=85_000,
            equity_history=[100_000, 85_000],
        )
        result = agent.decide_with_risk(vector, portfolio)
        assert result is None

    def test_decide_with_risk_passes_healthy_portfolio(self):
        """decide_with_risk() retourne un ordre si portfolio sain."""
        agent = make_decision_agent_mock(LONG_RESPONSE)
        vector = SignalVector(symbol="AAPL", n_bars=500, data_quality=1.0)

        portfolio = Portfolio(cash=100_000, equity_history=[95_000, 98_000, 100_000])
        result = agent.decide_with_risk(vector, portfolio, recent_returns=normal_returns())
        assert result is not None
        assert result.direction == "LONG"

    def test_decide_with_risk_custom_limits(self):
        """decide_with_risk() respecte les limites custom."""
        agent = make_decision_agent_mock(LONG_RESPONSE)
        vector = SignalVector(symbol="AAPL", n_bars=500, data_quality=1.0)

        # Portfolio sain mais limit ultra-stricte drawdown = 1%
        portfolio = Portfolio(cash=98_000, equity_history=[100_000, 98_000])
        strict_limits = RiskLimits(max_drawdown=0.01)
        result = agent.decide_with_risk(vector, portfolio, limits=strict_limits)
        assert result is None
