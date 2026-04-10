"""
tests/test_decision_logic.py
-----------------------------
Tests pour TemporalDecisionEngine — donnees synthetiques, pas d'API.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from execution.decision_agent import TradeOrder
from execution.decision_logic import (
    DecisionParams,
    SignalState,
    TemporalDecisionEngine,
)
from signals.aggregator import SignalVector

# ---------------------------------------------------------------------------
# Helpers — synthetic data builders
# ---------------------------------------------------------------------------


def _make_vector(
    symbol: str = "AAPL",
    momentum_composite: float | None = 0.5,
    momentum_3m: float | None = 0.3,
    momentum_12m: float | None = 0.4,
    value: float | None = 0.2,
    quality: float | None = 0.3,
    low_volatility: float | None = 0.1,
    ml_prediction: float | None = 0.4,
    agent_bias: float | None = 0.3,
    mirofish_sentiment: float | None = 0.3,
) -> SignalVector:
    """Build a synthetic SignalVector with controllable fields."""
    return SignalVector(
        symbol=symbol,
        momentum_composite=momentum_composite,
        momentum_3m=momentum_3m,
        momentum_12m=momentum_12m,
        value=value,
        quality=quality,
        low_volatility=low_volatility,
        ml_prediction=ml_prediction,
        agent_bias=agent_bias,
        mirofish_sentiment=mirofish_sentiment,
        n_bars=300,
        data_quality=1.0,
    )


def _make_order(
    symbol: str = "AAPL",
    direction: str = "LONG",
    confidence: float = 0.85,
    size_pct: float = 0.03,
) -> TradeOrder:
    """Build a synthetic TradeOrder."""
    return TradeOrder(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        size_pct=size_pct,
        rationale="test order",
    )


def _make_engine(
    params: DecisionParams | None = None,
) -> TemporalDecisionEngine:
    """Build an engine with a temp state file (no disk pollution)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    state_file = Path(tmp.name)
    state_file.unlink(missing_ok=True)  # start with no file
    return TemporalDecisionEngine(
        params=params or DecisionParams(),
        state_file=state_file,
    )


# A Wednesday at 11:00 — inside trading window
WEEKDAY_11H = datetime(2026, 4, 8, 11, 0, 0)
# A Wednesday at 8:00 — outside trading window
WEEKDAY_08H = datetime(2026, 4, 8, 8, 0, 0)
# A Saturday at 11:00
SATURDAY_11H = datetime(2026, 4, 11, 11, 0, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_initial_state_is_neutral(self) -> None:
        """Nouvel engine -> etat NEUTRAL pour tout symbole."""
        engine = _make_engine()
        # Process with weak signal so it stays NEUTRAL
        action = engine.process(
            "AAPL",
            _make_vector(momentum_composite=0.01, value=0.0, quality=0.0),
            _make_order(direction="FLAT", confidence=0.2),
            now=WEEKDAY_11H,
        )
        assert action.state_after == "NEUTRAL"
        assert action.should_trade is False


class TestNeutralToWatching:
    def test_neutral_to_watching_on_strong_signal(self) -> None:
        """Signal fort (composite>0.3, 3+ signaux) -> WATCHING."""
        engine = _make_engine()
        vector = _make_vector()  # strong defaults
        order = _make_order()
        action = engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert action.state_before == "NEUTRAL"
        assert action.state_after == "WATCHING"
        assert action.should_trade is False


class TestWatchingToNeutral:
    def test_watching_to_neutral_if_signal_disappears(self) -> None:
        """Signal disparait en WATCHING -> reset NEUTRAL."""
        engine = _make_engine()
        # First: strong signal -> WATCHING
        engine.process("AAPL", _make_vector(), _make_order(), now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "WATCHING"

        # Second: weak signal -> should reset to NEUTRAL
        weak_vector = _make_vector(
            momentum_composite=0.01,
            value=-0.5,
            quality=-0.5,
            ml_prediction=-0.5,
            agent_bias=-0.5,
            momentum_3m=0.3,
            momentum_12m=0.4,
        )
        action = engine.process("AAPL", weak_vector, _make_order(), now=WEEKDAY_11H)
        assert action.state_after == "NEUTRAL"
        assert action.should_trade is False


class TestConfirmationDays:
    def test_confirmation_days_increment(self) -> None:
        """Signal coherent 3 jours -> confirmation_days=3."""
        engine = _make_engine(DecisionParams(min_confirmation_days=5))
        vector = _make_vector()
        order = _make_order()

        # Day 1: NEUTRAL -> WATCHING (confirmation_days=1)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].confirmation_days == 1

        # Day 2: still WATCHING (confirmation_days=2)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].confirmation_days == 2

        # Day 3: still WATCHING (confirmation_days=3)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].confirmation_days == 3


class TestWatchingToConfirmed:
    def test_watching_to_confirmed_after_n_days(self) -> None:
        """Apres min_confirmation_days -> CONFIRMED."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Day 1: NEUTRAL -> WATCHING
        action = engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert action.state_after == "WATCHING"

        # Day 2: WATCHING -> CONFIRMED (confirmation_days reaches 2)
        action = engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert action.state_after == "CONFIRMED"


class TestConfirmedToEntered:
    def test_confirmed_to_entered_in_trading_window(self) -> None:
        """CONFIRMED + fenetre trading -> should_trade=True."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Get to CONFIRMED
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "CONFIRMED"

        # Process in trading window -> ENTERED with should_trade=True
        action = engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert action.state_after == "ENTERED"
        assert action.should_trade is True
        assert action.trade_order is not None
        assert action.trade_order.direction == "LONG"


class TestTradingWindow:
    def test_no_trade_outside_trading_window(self) -> None:
        """CONFIRMED mais heure=8h -> should_trade=False."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Get to CONFIRMED
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "CONFIRMED"

        # Process outside window -> stays CONFIRMED, no trade
        action = engine.process("AAPL", vector, order, now=WEEKDAY_08H)
        assert action.state_after == "CONFIRMED"
        assert action.should_trade is False

    def test_no_trade_weekend(self) -> None:
        """CONFIRMED mais samedi -> should_trade=False."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Get to CONFIRMED
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "CONFIRMED"

        # Process on Saturday -> stays CONFIRMED
        action = engine.process("AAPL", vector, order, now=SATURDAY_11H)
        assert action.state_after == "CONFIRMED"
        assert action.should_trade is False


class TestProgressiveSizing:
    def test_progressive_sizing(self) -> None:
        """Jour 1=2%, Jour 2=3%, Jour 3=4%, Jour 4+=5%."""
        engine = _make_engine(DecisionParams(min_confirmation_days=1))
        state = SignalState(symbol="TEST", confirmation_days=1)

        # min_confirmation_days=1, so extra_days = conf_days - min_conf_days
        # conf_days=1 -> extra=0 -> 2%
        state.confirmation_days = 1
        assert engine._compute_size(state) == pytest.approx(0.02)

        # conf_days=2 -> extra=1 -> 2% + 1% = 3%
        state.confirmation_days = 2
        assert engine._compute_size(state) == pytest.approx(0.03)

        # conf_days=3 -> extra=2 -> 2% + 2% = 4%
        state.confirmation_days = 3
        assert engine._compute_size(state) == pytest.approx(0.04)

        # conf_days=4 -> extra=3 -> 2% + 3% = 5% (capped)
        state.confirmation_days = 4
        assert engine._compute_size(state) == pytest.approx(0.05)

        # conf_days=10 -> still capped at 5%
        state.confirmation_days = 10
        assert engine._compute_size(state) == pytest.approx(0.05)


class TestMomentumAlignment:
    def test_momentum_alignment_required(self) -> None:
        """mom_3m>0 mais mom_12m<0 -> pas de confirmation."""
        engine = _make_engine(DecisionParams(require_momentum_alignment=True))
        # Misaligned momentum: 3m positive, 12m negative
        vector = _make_vector(
            momentum_3m=0.5,
            momentum_12m=-0.3,
        )
        order = _make_order()
        action = engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        # Should stay NEUTRAL because momentum not aligned
        assert action.state_after == "NEUTRAL"


class TestMinSignalsAligned:
    def test_min_signals_aligned_3(self) -> None:
        """Moins de 3 signaux alignes -> pas de WATCHING."""
        engine = _make_engine(DecisionParams(min_signals_aligned=3))
        # Only 2 aligned signals (value and quality), rest are None or weak
        vector = _make_vector(
            momentum_composite=0.5,  # composite is high
            momentum_3m=0.3,
            momentum_12m=0.4,
            value=0.1,     # aligned (>0 for LONG)
            quality=0.1,   # aligned (>0 for LONG)
            ml_prediction=None,
            agent_bias=None,
            mirofish_sentiment=None,
            low_volatility=None,
        )
        order = _make_order()
        action = engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        # Only value + quality = 2 aligned, need 3 -> stays NEUTRAL
        assert action.state_after == "NEUTRAL"


class TestPersistence:
    def test_state_persists_to_json(self) -> None:
        """states sauves dans JSON -> recharges au prochain run."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        state_file = Path(tmp.name)
        state_file.unlink(missing_ok=True)

        # Engine 1: process to get WATCHING state
        engine1 = TemporalDecisionEngine(
            params=DecisionParams(),
            state_file=state_file,
        )
        engine1.process("AAPL", _make_vector(), _make_order(), now=WEEKDAY_11H)
        assert engine1.states["AAPL"].state == "WATCHING"
        engine1._save_states()

        # Engine 2: reload from same file
        engine2 = TemporalDecisionEngine(
            params=DecisionParams(),
            state_file=state_file,
        )
        assert "AAPL" in engine2.states
        assert engine2.states["AAPL"].state == "WATCHING"
        assert engine2.states["AAPL"].direction == "LONG"

        # Cleanup
        state_file.unlink(missing_ok=True)


class TestReset:
    def test_reset_single_symbol(self) -> None:
        """reset('AAPL') -> AAPL=NEUTRAL, autres inchanges."""
        engine = _make_engine()
        vector = _make_vector()
        order = _make_order()

        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("MSFT", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "WATCHING"
        assert engine.states["MSFT"].state == "WATCHING"

        engine.reset("AAPL")
        assert engine.states["AAPL"].state == "NEUTRAL"
        assert engine.states["MSFT"].state == "WATCHING"

    def test_reset_all(self) -> None:
        """reset() -> tous NEUTRAL."""
        engine = _make_engine()
        vector = _make_vector()
        order = _make_order()

        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("MSFT", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "WATCHING"
        assert engine.states["MSFT"].state == "WATCHING"

        engine.reset()
        assert engine.states["AAPL"].state == "NEUTRAL"
        assert engine.states["MSFT"].state == "NEUTRAL"


class TestWatchingExpiry:
    def test_watching_expires_after_max_days(self) -> None:
        """WATCHING depuis max_watching_days -> reset NEUTRAL."""
        engine = _make_engine(DecisionParams(
            max_watching_days=5,
            min_confirmation_days=10,  # very high so we never confirm
        ))
        vector = _make_vector()
        order = _make_order(confidence=0.5)  # low confidence, won't confirm

        # Day 0: NEUTRAL -> WATCHING
        day0 = datetime(2026, 4, 1, 11, 0, 0)
        engine.process("AAPL", vector, order, now=day0)
        assert engine.states["AAPL"].state == "WATCHING"

        # Day 6 (>5 max_watching_days): should expire
        day6 = datetime(2026, 4, 7, 11, 0, 0)
        action = engine.process("AAPL", vector, order, now=day6)
        assert action.state_after == "NEUTRAL"
        assert "max watching days" in action.reason


class TestSignalReversal:
    def test_entered_to_exited_on_signal_reversal(self) -> None:
        """Position LONG + composite fort negatif -> EXITED."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Get to ENTERED
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "ENTERED"

        # Now send strong negative signal
        bearish_vector = _make_vector(
            momentum_composite=-0.8,
            value=-0.5,
            quality=-0.5,
            ml_prediction=-0.5,
            momentum_3m=-0.3,
            momentum_12m=-0.4,
        )
        action = engine.process("AAPL", bearish_vector, order, now=WEEKDAY_11H)
        assert action.state_after == "EXITED"
        assert action.should_trade is True
        assert action.trade_order is not None
        assert action.trade_order.direction == "SHORT"  # close LONG


class TestStatusDataFrame:
    def test_status_returns_dataframe(self) -> None:
        """status() -> DataFrame avec toutes les colonnes."""
        engine = _make_engine()
        vector = _make_vector()
        order = _make_order()

        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("MSFT", vector, order, now=WEEKDAY_11H)

        df = engine.status()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        expected_cols = {"Symbol", "State", "Direction", "Conf Days", "Composite", "ML", "Aligned"}
        assert set(df.columns) == expected_cols

    def test_status_empty_engine(self) -> None:
        """status() on empty engine -> empty DataFrame with correct columns."""
        engine = _make_engine()
        df = engine.status()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "Symbol" in df.columns


class TestFlatOrder:
    def test_flat_order_does_not_trigger_trade(self) -> None:
        """Claude dit FLAT -> should_trade=False meme en CONFIRMED."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Get to CONFIRMED
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "CONFIRMED"

        # Process with FLAT order -> should not trade
        flat_order = _make_order(direction="FLAT", confidence=0.3)
        action = engine.process("AAPL", vector, flat_order, now=WEEKDAY_11H)
        assert action.should_trade is False
        assert "FLAT" in action.reason


class TestProcessUniverse:
    def test_process_universe_returns_only_tradeable(self) -> None:
        """process_universe returns only actions with should_trade=True."""
        engine = _make_engine(DecisionParams(min_confirmation_days=2))
        vector = _make_vector()
        order = _make_order(confidence=0.85)

        # Get AAPL to CONFIRMED then process universe
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        engine.process("AAPL", vector, order, now=WEEKDAY_11H)
        assert engine.states["AAPL"].state == "CONFIRMED"

        signals: dict[str, tuple[SignalVector, TradeOrder | None]] = {
            "AAPL": (vector, order),   # CONFIRMED -> ENTERED (trade)
            "MSFT": (vector, order),   # NEUTRAL -> WATCHING (no trade)
        }
        actions = engine.process_universe(signals, now=WEEKDAY_11H)
        # Only AAPL should trade
        assert len(actions) == 1
        assert actions[0].symbol == "AAPL"
        assert actions[0].should_trade is True
