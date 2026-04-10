"""
execution/decision_logic.py
----------------------------
Logique de decision temporelle pour le pipeline QuantAI.

Au lieu de trader immediatement sur un signal, le TemporalDecisionEngine
impose une confirmation sur plusieurs jours, exige la convergence de
plusieurs sources de signal, respecte des fenetres de trading, et entre
progressivement dans une position.

Machine d'etat par symbole :
    NEUTRAL -> WATCHING -> CONFIRMED -> ENTERED -> EXITED -> NEUTRAL

Usage :
    engine = TemporalDecisionEngine()

    # Chaque jour / chaque run :
    action = engine.process(symbol="AAPL", vector=vector, order=claude_order)

    if action.should_trade:
        router.submit(action.trade_order, symbol="AAPL")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger

from execution.decision_agent import Direction, TradeOrder
from signals.aggregator import SignalVector

# ---------------------------------------------------------------------------
# Decision parameters
# ---------------------------------------------------------------------------


@dataclass
class DecisionParams:
    """Parametres de la logique de decision temporelle."""

    # Confirmation temporelle
    min_confirmation_days: int = 2      # signal doit tenir N jours avant d'agir
    max_watching_days: int = 5          # si pas confirme en 5j -> reset NEUTRAL

    # Convergence des signaux
    min_composite_score: float = 0.30   # composite_score minimum pour agir
    min_confidence: float = 0.70        # confidence Claude minimum
    min_signals_aligned: int = 3        # nb de signaux qui doivent pointer dans le meme sens

    # Fenetre de trading (heure locale)
    trading_window_start: int = 10      # 10h00
    trading_window_end: int = 15        # 15h00

    # Position sizing progressif
    initial_size_pct: float = 0.02      # 2% au premier signal
    max_size_pct: float = 0.05          # 5% maximum total
    scale_in_days: int = 3              # ajouter 1% par jour de confirmation supplementaire

    # Multi-timeframe
    require_momentum_alignment: bool = True  # mom_3m ET mom_12m doivent pointer pareil


# ---------------------------------------------------------------------------
# Signal state per symbol
# ---------------------------------------------------------------------------


@dataclass
class SignalState:
    """Etat d'un symbole dans la machine d'etat temporelle."""

    symbol: str
    state: str = "NEUTRAL"              # NEUTRAL | WATCHING | CONFIRMED | ENTERED | EXITED
    direction: str | None = None        # "LONG" | "SHORT"
    confirmation_days: int = 0          # nb de jours consecutifs ou le signal est coherent
    first_signal_date: str | None = None
    last_signal_date: str | None = None
    last_composite: float | None = None
    last_ml: float | None = None
    last_confidence: float | None = None
    signals_history: list[dict] = field(default_factory=list)
    current_size_pct: float = 0.0       # taille actuelle de la position
    entry_date: str | None = None
    exit_date: str | None = None        # date de sortie pour cooling off

    def days_watching(self) -> int:
        if self.first_signal_date is None:
            return 0
        first = datetime.fromisoformat(self.first_signal_date)
        return (datetime.now() - first).days

    def to_dict(self) -> dict[str, Any]:
        """Serialisable JSON pour persistance."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Decision action returned by process()
# ---------------------------------------------------------------------------


@dataclass
class DecisionAction:
    """Resultat d'un appel a process() — indique si on doit trader."""

    symbol: str
    state_before: str
    state_after: str
    should_trade: bool = False
    trade_order: TradeOrder | None = None
    reason: str = ""
    n_signals_aligned: int = 0

    def __str__(self) -> str:
        if self.should_trade and self.trade_order is not None:
            trade = f"-> TRADE {self.trade_order.direction}"
        else:
            trade = "-> WAIT"
        return (
            f"[{self.symbol}] {self.state_before}->{self.state_after} "
            f"{trade} ({self.reason})"
        )


# ---------------------------------------------------------------------------
# Temporal Decision Engine
# ---------------------------------------------------------------------------


class TemporalDecisionEngine:
    """
    Gere la logique de decision temporelle pour un univers de symboles.

    Persiste l'etat dans execution/signal_states.json entre les runs.
    Chaque appel a process() fait avancer la machine d'etat.

    Usage :
        engine = TemporalDecisionEngine()

        # Chaque jour / chaque run :
        action = engine.process(symbol="AAPL", vector=vector, order=claude_order)

        if action.should_trade:
            router.submit(action.trade_order, symbol="AAPL")
    """

    DEFAULT_STATE_FILE = Path("execution/signal_states.json")

    def __init__(
        self,
        params: DecisionParams | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.params = params or DecisionParams()
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self.states: dict[str, SignalState] = {}
        self._load_states()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load_states(self) -> None:
        """Charge les etats depuis le fichier JSON."""
        if not self.state_file.exists():
            return
        try:
            raw = self.state_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            for symbol, state_dict in data.items():
                self.states[symbol] = SignalState(**state_dict)
            logger.info(f"Loaded {len(self.states)} signal states from {self.state_file}")
        except Exception as e:
            logger.warning(f"Could not load signal states ({e}) — starting fresh")
            self.states = {}

    def _save_states(self) -> None:
        """Persiste les etats dans le fichier JSON."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {symbol: state.to_dict() for symbol, state in self.states.items()}
            self.state_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Could not save signal states: {e}")

    # -----------------------------------------------------------------------
    # Signal analysis helpers
    # -----------------------------------------------------------------------

    def _count_aligned_signals(self, vector: SignalVector, direction: str) -> int:
        """
        Compte combien de signaux pointent dans la meme direction.

        Signaux verifies :
        - momentum_composite > 0.2 (LONG) ou < -0.2 (SHORT)
        - ml_prediction > 0.2 (LONG) ou < -0.2 (SHORT)
        - value > 0 (LONG) ou < 0 (SHORT)
        - quality > 0 (LONG) ou < 0 (SHORT)
        - low_volatility > 0 (LONG) — faible vol favorable aux LONG
        - mirofish_sentiment > 0.2 (LONG) ou < -0.2 (SHORT)
        - agent_bias > 0.2 (LONG) ou < -0.2 (SHORT)

        Retourne le nombre de signaux alignes avec direction.
        """
        count = 0
        is_long = direction == "LONG"

        # momentum_composite
        if vector.momentum_composite is not None:
            if (is_long and vector.momentum_composite > 0.2) or (
                not is_long and vector.momentum_composite < -0.2
            ):
                count += 1

        # ml_prediction
        if vector.ml_prediction is not None:
            if (is_long and vector.ml_prediction > 0.2) or (
                not is_long and vector.ml_prediction < -0.2
            ):
                count += 1

        # value
        if vector.value is not None:
            if (is_long and vector.value > 0) or (not is_long and vector.value < 0):
                count += 1

        # quality
        if vector.quality is not None:
            if (is_long and vector.quality > 0) or (not is_long and vector.quality < 0):
                count += 1

        # low_volatility — faible vol favorable aux LONG seulement
        if vector.low_volatility is not None:
            if is_long and vector.low_volatility > 0:
                count += 1

        # mirofish_sentiment
        if vector.mirofish_sentiment is not None:
            if (is_long and vector.mirofish_sentiment > 0.2) or (
                not is_long and vector.mirofish_sentiment < -0.2
            ):
                count += 1

        # agent_bias
        if vector.agent_bias is not None:
            if (is_long and vector.agent_bias > 0.2) or (
                not is_long and vector.agent_bias < -0.2
            ):
                count += 1

        return count

    def _is_in_trading_window(self, now: datetime | None = None) -> bool:
        """Verifie qu'on est dans la fenetre de trading autorisee."""
        if now is None:
            now = datetime.now()
        # Ne pas trader le weekend
        if now.weekday() >= 5:
            return False
        hour = now.hour
        return self.params.trading_window_start <= hour < self.params.trading_window_end

    def _check_momentum_alignment(self, vector: SignalVector) -> bool:
        """
        Verifie l'alignement multi-timeframe :
        mom_3m et mom_12m doivent pointer dans le meme sens.
        """
        if not self.params.require_momentum_alignment:
            return True
        mom3 = vector.momentum_3m
        mom12 = vector.momentum_12m
        if mom3 is None or mom12 is None:
            return False
        return (mom3 > 0 and mom12 > 0) or (mom3 < 0 and mom12 < 0)

    def _compute_size(self, state: SignalState) -> float:
        """
        Position sizing progressif base sur les jours de confirmation.

        Jour 1 : initial_size_pct (2%)
        Jour 2 : initial_size_pct + 1% (3%)
        Jour 3 : initial_size_pct + 2% (4%)
        Jour 4+ : max_size_pct (5%)
        """
        extra_days = max(0, state.confirmation_days - self.params.min_confirmation_days)
        extra = extra_days * 0.01
        return min(self.params.initial_size_pct + extra, self.params.max_size_pct)

    def _infer_direction(self, vector: SignalVector, order: TradeOrder | None) -> str | None:
        """Infer direction from order or composite score."""
        if order is not None and order.direction in ("LONG", "SHORT"):
            return order.direction
        cs = vector.composite_score
        if cs is not None:
            if cs > 0:
                return "LONG"
            if cs < 0:
                return "SHORT"
        return None

    # -----------------------------------------------------------------------
    # Main state machine
    # -----------------------------------------------------------------------

    def process(
        self,
        symbol: str,
        vector: SignalVector,
        order: TradeOrder | None = None,
        now: datetime | None = None,
    ) -> DecisionAction:
        """
        Machine d'etat principale. Fait avancer l'etat du symbole.

        Logique :

        NEUTRAL :
            Si composite fort + signaux alignes + momentum aligne
            -> passer en WATCHING, enregistrer le premier signal

        WATCHING :
            Si signal toujours coherent
            -> incrementer confirmation_days
            Si confirmation_days >= min_confirmation_days ET fenetre trading
            -> passer en CONFIRMED
            Si signal disparait ou contradictoire
            -> reset NEUTRAL
            Si days_watching > max_watching_days
            -> reset NEUTRAL (signal trop lent)

        CONFIRMED :
            Si dans fenetre de trading
            -> should_trade=True, taille = _compute_size()
            -> passer en ENTERED

        ENTERED :
            Surveiller la position
            Si signal s'inverse franchement (composite change de signe fortement)
            -> should_trade=True avec direction inverse (cloture)
            -> passer en EXITED

        EXITED :
            -> reset NEUTRAL apres 1 jour de cooling off
        """
        if now is None:
            now = datetime.now()

        if symbol not in self.states:
            self.states[symbol] = SignalState(symbol=symbol)

        state = self.states[symbol]
        state_before = state.state
        composite = vector.composite_score
        direction = self._infer_direction(vector, order)
        confidence = order.confidence if order is not None else 0.0
        n_aligned = self._count_aligned_signals(vector, direction) if direction else 0

        # Record signal history
        state.signals_history.append({
            "date": now.isoformat(),
            "composite": composite,
            "direction": direction,
            "confidence": confidence,
            "n_aligned": n_aligned,
        })
        # Keep last 30 entries
        if len(state.signals_history) > 30:
            state.signals_history = state.signals_history[-30:]

        state.last_composite = composite
        state.last_ml = vector.ml_prediction
        state.last_confidence = confidence

        # ----- EXITED -----
        if state.state == "EXITED":
            # Cooling off: reset to NEUTRAL after 1 day
            if state.exit_date is not None:
                exit_dt = datetime.fromisoformat(state.exit_date)
                if (now - exit_dt).days >= 1:
                    self._reset_state(state)
                    logger.info(f"[{symbol}] EXITED->NEUTRAL (cooling off complete)")
                    return DecisionAction(
                        symbol=symbol,
                        state_before=state_before,
                        state_after="NEUTRAL",
                        reason="cooling off complete",
                        n_signals_aligned=n_aligned,
                    )
            return DecisionAction(
                symbol=symbol,
                state_before=state_before,
                state_after="EXITED",
                reason="cooling off",
                n_signals_aligned=n_aligned,
            )

        # ----- ENTERED -----
        if state.state == "ENTERED":
            # Check for signal reversal
            if composite is not None and state.direction is not None:
                reversal = False
                if state.direction == "LONG" and composite < -0.2:
                    reversal = True
                elif state.direction == "SHORT" and composite > 0.2:
                    reversal = True

                if reversal:
                    close_dir: Direction = "SHORT" if state.direction == "LONG" else "LONG"
                    close_order = TradeOrder(
                        symbol=symbol,
                        direction=close_dir,
                        confidence=1.0,
                        size_pct=state.current_size_pct,
                        rationale=f"Signal reversal: closing {state.direction} position",
                    )
                    state.state = "EXITED"
                    state.exit_date = now.isoformat()
                    logger.info(f"[{symbol}] ENTERED->EXITED (signal reversal, composite={composite:+.3f})")
                    return DecisionAction(
                        symbol=symbol,
                        state_before=state_before,
                        state_after="EXITED",
                        should_trade=True,
                        trade_order=close_order,
                        reason=f"signal reversal (composite={composite:+.3f})",
                        n_signals_aligned=n_aligned,
                    )

            return DecisionAction(
                symbol=symbol,
                state_before=state_before,
                state_after="ENTERED",
                reason="position held",
                n_signals_aligned=n_aligned,
            )

        # ----- CONFIRMED -----
        if state.state == "CONFIRMED":
            # If order is FLAT, do not trade
            if order is not None and order.direction == "FLAT":
                return DecisionAction(
                    symbol=symbol,
                    state_before=state_before,
                    state_after="CONFIRMED",
                    reason="Claude says FLAT — waiting",
                    n_signals_aligned=n_aligned,
                )

            if self._is_in_trading_window(now):
                size = self._compute_size(state)
                entry_dir = cast(Direction, state.direction or "LONG")
                trade_order = TradeOrder(
                    symbol=symbol,
                    direction=entry_dir,
                    confidence=confidence,
                    size_pct=size,
                    stop_loss=order.stop_loss if order else 0.02,
                    take_profit=order.take_profit if order else 0.04,
                    rationale=f"Confirmed after {state.confirmation_days}d, {n_aligned} signals aligned",
                )
                state.state = "ENTERED"
                state.current_size_pct = size
                state.entry_date = now.isoformat()
                logger.info(
                    f"[{symbol}] CONFIRMED->ENTERED (size={size:.1%}, "
                    f"conf_days={state.confirmation_days})"
                )
                return DecisionAction(
                    symbol=symbol,
                    state_before=state_before,
                    state_after="ENTERED",
                    should_trade=True,
                    trade_order=trade_order,
                    reason=f"confirmed {state.confirmation_days}d, entering position",
                    n_signals_aligned=n_aligned,
                )
            else:
                return DecisionAction(
                    symbol=symbol,
                    state_before=state_before,
                    state_after="CONFIRMED",
                    reason="outside trading window",
                    n_signals_aligned=n_aligned,
                )

        # ----- WATCHING -----
        if state.state == "WATCHING":
            # Check if signal disappeared or contradicts
            signal_consistent = self._is_signal_consistent(state, vector, direction)

            if not signal_consistent:
                self._reset_state(state)
                logger.info(f"[{symbol}] WATCHING->NEUTRAL (signal lost)")
                return DecisionAction(
                    symbol=symbol,
                    state_before=state_before,
                    state_after="NEUTRAL",
                    reason="signal disappeared or contradicted",
                    n_signals_aligned=n_aligned,
                )

            # Check max watching days
            if state.days_watching() > self.params.max_watching_days:
                self._reset_state(state)
                logger.info(f"[{symbol}] WATCHING->NEUTRAL (max watching days exceeded)")
                return DecisionAction(
                    symbol=symbol,
                    state_before=state_before,
                    state_after="NEUTRAL",
                    reason="max watching days exceeded",
                    n_signals_aligned=n_aligned,
                )

            # Signal still consistent — increment confirmation
            state.confirmation_days += 1
            state.last_signal_date = now.isoformat()

            if (
                state.confirmation_days >= self.params.min_confirmation_days
                and confidence >= self.params.min_confidence
            ):
                state.state = "CONFIRMED"
                logger.info(
                    f"[{symbol}] WATCHING->CONFIRMED "
                    f"(conf_days={state.confirmation_days}, confidence={confidence:.2f})"
                )
                return DecisionAction(
                    symbol=symbol,
                    state_before=state_before,
                    state_after="CONFIRMED",
                    reason=f"confirmed after {state.confirmation_days} days",
                    n_signals_aligned=n_aligned,
                )

            return DecisionAction(
                symbol=symbol,
                state_before=state_before,
                state_after="WATCHING",
                reason=f"watching day {state.confirmation_days}/{self.params.min_confirmation_days}",
                n_signals_aligned=n_aligned,
            )

        # ----- NEUTRAL -----
        # Check if we should start watching
        if (
            composite is not None
            and abs(composite) >= self.params.min_composite_score
            and direction is not None
            and n_aligned >= self.params.min_signals_aligned
            and self._check_momentum_alignment(vector)
        ):
            state.state = "WATCHING"
            state.direction = direction
            state.confirmation_days = 1
            state.first_signal_date = now.isoformat()
            state.last_signal_date = now.isoformat()
            logger.info(
                f"[{symbol}] NEUTRAL->WATCHING "
                f"(direction={direction}, composite={composite:+.3f}, aligned={n_aligned})"
            )
            return DecisionAction(
                symbol=symbol,
                state_before=state_before,
                state_after="WATCHING",
                reason=f"strong signal detected (composite={composite:+.3f}, {n_aligned} aligned)",
                n_signals_aligned=n_aligned,
            )

        return DecisionAction(
            symbol=symbol,
            state_before=state_before,
            state_after="NEUTRAL",
            reason="no actionable signal",
            n_signals_aligned=n_aligned,
        )

    def _is_signal_consistent(
        self,
        state: SignalState,
        vector: SignalVector,
        direction: str | None,
    ) -> bool:
        """Check if the current signal is consistent with the watching state."""
        if direction is None:
            return False
        if direction != state.direction:
            return False
        composite = vector.composite_score
        if composite is None:
            return False
        if abs(composite) < self.params.min_composite_score:
            return False
        n_aligned = self._count_aligned_signals(vector, direction)
        if n_aligned < self.params.min_signals_aligned:
            return False
        if not self._check_momentum_alignment(vector):
            return False
        return True

    def _reset_state(self, state: SignalState) -> None:
        """Reset a signal state to NEUTRAL."""
        state.state = "NEUTRAL"
        state.direction = None
        state.confirmation_days = 0
        state.first_signal_date = None
        state.last_signal_date = None
        state.current_size_pct = 0.0
        state.entry_date = None
        state.exit_date = None

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------

    def process_universe(
        self,
        signals: dict[str, tuple[SignalVector, TradeOrder | None]],
        now: datetime | None = None,
    ) -> list[DecisionAction]:
        """
        Traite tout l'univers d'un coup.
        Retourne uniquement les actions avec should_trade=True.
        """
        actions = []
        for symbol, (vector, order) in signals.items():
            action = self.process(symbol, vector, order, now=now)
            actions.append(action)
            logger.info(str(action))
        self._save_states()
        return [a for a in actions if a.should_trade]

    # -----------------------------------------------------------------------
    # Status & reset
    # -----------------------------------------------------------------------

    def status(self) -> pd.DataFrame:
        """
        Affiche l'etat de tous les symboles suivis.

        Columns: Symbol, State, Direction, Conf Days, Composite, ML, Aligned
        """
        if not self.states:
            return pd.DataFrame(
                columns=["Symbol", "State", "Direction", "Conf Days", "Composite", "ML", "Aligned"]
            )

        rows = []
        for symbol, s in self.states.items():
            n_aligned = 0
            if s.signals_history:
                last = s.signals_history[-1]
                n_aligned = last.get("n_aligned", 0)

            rows.append({
                "Symbol": symbol,
                "State": s.state,
                "Direction": s.direction or "",
                "Conf Days": s.confirmation_days,
                "Composite": f"{s.last_composite:+.3f}" if s.last_composite is not None else "N/A",
                "ML": f"{s.last_ml:+.3f}" if s.last_ml is not None else "N/A",
                "Aligned": f"{n_aligned}/7",
            })
        return pd.DataFrame(rows)

    def reset(self, symbol: str | None = None) -> None:
        """Reset l'etat d'un symbole ou de tous."""
        if symbol is not None:
            if symbol in self.states:
                self._reset_state(self.states[symbol])
                logger.info(f"Reset {symbol} -> NEUTRAL")
        else:
            for s in self.states.values():
                self._reset_state(s)
            logger.info("Reset all symbols -> NEUTRAL")
        self._save_states()
