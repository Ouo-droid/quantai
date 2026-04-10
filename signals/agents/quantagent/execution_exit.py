"""
Execution & Exits Layer (Layer 4b) — sits between Decision and Risk Global.

Separates the decision to enter from the exit plan:
- Entry validation: confirms the decision meets minimum thresholds
- Stop-loss via ATR or market structure
- Take-profit / trailing stop logic
- Time-based exit rules
- R-multiple journaling: tracks risk/reward for every trade

Usage:
    manager = ExecutionExitManager()
    exit_plan = manager.process(state, decision_output)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ATR calculator
# ---------------------------------------------------------------------------


def compute_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float]:
    """
    Compute Average True Range (ATR) from OHLC data.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = SMA of True Range over *period* bars.

    Returns a list the same length as the input; the first *period* values
    are partial averages (or 0.0 when insufficient data).
    """
    n = len(highs)
    if n == 0 or n != len(lows) or n != len(closes):
        return []

    true_ranges: list[float] = []
    for i in range(n):
        hl = highs[i] - lows[i]
        if i == 0:
            true_ranges.append(hl)
        else:
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            true_ranges.append(max(hl, hc, lc))

    # Simple moving average of true ranges
    atr_values: list[float] = []
    running_sum = 0.0
    for i, tr in enumerate(true_ranges):
        running_sum += tr
        if i < period:
            atr_values.append(running_sum / (i + 1))
        else:
            running_sum -= true_ranges[i - period]
            atr_values.append(running_sum / period)

    return atr_values


# ---------------------------------------------------------------------------
# Entry validator
# ---------------------------------------------------------------------------


@dataclass
class EntryValidationResult:
    """Result of entry validation."""

    is_valid: bool
    direction: str  # LONG | SHORT | FLAT
    confidence: float
    risk_reward_ratio: float
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntryValidator:
    """
    Validates whether a trade decision should be executed.

    Checks:
    - Minimum confidence threshold
    - Valid direction (LONG or SHORT, not FLAT)
    - Minimum risk-reward ratio
    - Coherence between reports (optional)
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_risk_reward: float = 1.2,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward

    def validate(self, decision_data: dict[str, Any]) -> EntryValidationResult:
        """
        Validate a parsed decision dict.

        Expected keys: decision/direction, confidence, risk_reward_ratio.
        """
        direction = str(
            decision_data.get("decision", decision_data.get("direction", "FLAT"))
        )
        confidence = float(decision_data.get("confidence", 0.0))
        risk_reward = float(decision_data.get("risk_reward_ratio", 0.0))

        reasons: list[str] = []

        if direction not in ("LONG", "SHORT"):
            reasons.append(f"invalid_direction={direction}")

        if confidence < self.min_confidence:
            reasons.append(
                f"confidence={confidence:.2f}<min={self.min_confidence:.2f}"
            )

        if risk_reward < self.min_risk_reward:
            reasons.append(
                f"risk_reward={risk_reward:.2f}<min={self.min_risk_reward:.2f}"
            )

        return EntryValidationResult(
            is_valid=len(reasons) == 0,
            direction=direction,
            confidence=confidence,
            risk_reward_ratio=risk_reward,
            rejection_reasons=reasons,
        )


# ---------------------------------------------------------------------------
# Stop-loss calculator
# ---------------------------------------------------------------------------


@dataclass
class StopLossLevel:
    """Computed stop-loss level with method info."""

    method: str  # "atr" | "structure" | "fixed"
    stop_price: float
    distance_pct: float  # distance as % of entry price
    atr_multiple: float = 0.0  # only for ATR method

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StopLossCalculator:
    """
    Computes stop-loss using ATR and/or market structure.

    Priority:
    1. ATR-based stop (default 2x ATR below/above entry)
    2. Market structure (recent swing low for LONG, swing high for SHORT)
    3. Falls back to a fixed percentage if no price data available
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        fixed_stop_pct: float = 0.02,
        atr_period: int = 14,
    ) -> None:
        self.atr_multiplier = atr_multiplier
        self.fixed_stop_pct = fixed_stop_pct
        self.atr_period = atr_period

    def compute(
        self,
        direction: str,
        entry_price: float,
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        closes: list[float] | None = None,
    ) -> StopLossLevel:
        """
        Compute the best stop-loss level.

        Uses ATR if OHLC data is provided, otherwise falls back to fixed %.
        Also checks market structure (swing points) and returns the tighter stop.
        """
        atr_stop = self._atr_stop(direction, entry_price, highs, lows, closes)
        structure_stop = self._structure_stop(direction, entry_price, highs, lows)

        # Pick the tighter stop (closer to entry) among available methods
        candidates: list[StopLossLevel] = []
        if atr_stop is not None:
            candidates.append(atr_stop)
        if structure_stop is not None:
            candidates.append(structure_stop)

        if not candidates:
            # Fixed fallback
            if direction == "LONG":
                stop_price = entry_price * (1.0 - self.fixed_stop_pct)
            else:
                stop_price = entry_price * (1.0 + self.fixed_stop_pct)
            return StopLossLevel(
                method="fixed",
                stop_price=round(stop_price, 4),
                distance_pct=self.fixed_stop_pct,
            )

        # Tighter = smaller distance_pct
        return min(candidates, key=lambda s: s.distance_pct)

    def _atr_stop(
        self,
        direction: str,
        entry_price: float,
        highs: list[float] | None,
        lows: list[float] | None,
        closes: list[float] | None,
    ) -> StopLossLevel | None:
        if not highs or not lows or not closes:
            return None
        if len(highs) < 2:
            return None

        atr_values = compute_atr(highs, lows, closes, self.atr_period)
        if not atr_values:
            return None

        current_atr = atr_values[-1]
        stop_distance = current_atr * self.atr_multiplier

        if direction == "LONG":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        distance_pct = stop_distance / entry_price if entry_price > 0 else 0.0

        return StopLossLevel(
            method="atr",
            stop_price=round(stop_price, 4),
            distance_pct=round(distance_pct, 4),
            atr_multiple=self.atr_multiplier,
        )

    def _structure_stop(
        self,
        direction: str,
        entry_price: float,
        highs: list[float] | None,
        lows: list[float] | None,
    ) -> StopLossLevel | None:
        """Use recent swing low (LONG) or swing high (SHORT) as stop."""
        if direction == "LONG" and lows and len(lows) >= 3:
            # Swing low = recent minimum over last N bars
            lookback = min(len(lows), 20)
            swing_low = min(lows[-lookback:])
            if swing_low < entry_price:
                distance_pct = (entry_price - swing_low) / entry_price
                return StopLossLevel(
                    method="structure",
                    stop_price=round(swing_low, 4),
                    distance_pct=round(distance_pct, 4),
                )

        if direction == "SHORT" and highs and len(highs) >= 3:
            lookback = min(len(highs), 20)
            swing_high = max(highs[-lookback:])
            if swing_high > entry_price:
                distance_pct = (swing_high - entry_price) / entry_price
                return StopLossLevel(
                    method="structure",
                    stop_price=round(swing_high, 4),
                    distance_pct=round(distance_pct, 4),
                )

        return None


# ---------------------------------------------------------------------------
# Take-profit / trailing logic
# ---------------------------------------------------------------------------


@dataclass
class TakeProfitPlan:
    """Structured take-profit and trailing plan."""

    initial_target_price: float
    initial_target_pct: float  # distance as % of entry
    trailing_active: bool = False
    trailing_offset_pct: float = 0.0  # trail distance as %
    partial_targets: list[dict[str, float]] = field(default_factory=list)
    # e.g. [{"r_multiple": 1.5, "exit_pct": 0.5}, {"r_multiple": 2.5, "exit_pct": 1.0}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TakeProfitManager:
    """
    Computes take-profit targets and trailing stop logic.

    Features:
    - Initial TP based on R-multiple (risk unit = stop distance)
    - Partial exits at intermediate R-multiples
    - Trailing stop that activates once price reaches 1R profit
    """

    def __init__(
        self,
        default_r_target: float = 2.0,
        trailing_activation_r: float = 1.0,
        trailing_offset_pct: float = 0.01,
        partial_targets: list[dict[str, float]] | None = None,
    ) -> None:
        self.default_r_target = default_r_target
        self.trailing_activation_r = trailing_activation_r
        self.trailing_offset_pct = trailing_offset_pct
        self.partial_targets = partial_targets or [
            {"r_multiple": 1.5, "exit_pct": 0.50},
            {"r_multiple": 2.5, "exit_pct": 1.00},
        ]

    def compute(
        self,
        direction: str,
        entry_price: float,
        stop_distance: float,
    ) -> TakeProfitPlan:
        """
        Compute take-profit plan based on the risk unit (stop distance).

        Args:
            direction: LONG or SHORT
            entry_price: price at entry
            stop_distance: absolute distance from entry to stop-loss
        """
        risk_unit = abs(stop_distance) if stop_distance != 0 else entry_price * 0.02
        target_distance = risk_unit * self.default_r_target

        if direction == "LONG":
            target_price = entry_price + target_distance
        else:
            target_price = entry_price - target_distance

        target_pct = target_distance / entry_price if entry_price > 0 else 0.0

        return TakeProfitPlan(
            initial_target_price=round(target_price, 4),
            initial_target_pct=round(target_pct, 4),
            trailing_active=True,
            trailing_offset_pct=self.trailing_offset_pct,
            partial_targets=list(self.partial_targets),
        )

    def compute_trailing_stop(
        self,
        direction: str,
        highest_price: float,
        lowest_price: float,
    ) -> float:
        """
        Compute trailing stop price from the best price reached.

        For LONG: trail below the highest price.
        For SHORT: trail above the lowest price.
        """
        if direction == "LONG":
            return round(highest_price * (1.0 - self.trailing_offset_pct), 4)
        return round(lowest_price * (1.0 + self.trailing_offset_pct), 4)


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------


@dataclass
class TimeExitRule:
    """Time-based exit evaluation."""

    should_exit: bool
    reason: str
    bars_held: int
    max_bars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TimeBasedExit:
    """
    Enforces time-based exit rules.

    - Maximum holding period (in bars/candles)
    - Weekend risk reduction (exit before weekend if intraday)
    - Stale trade detection (no progress after N bars)
    """

    def __init__(
        self,
        max_holding_bars: int = 20,
        stale_bars: int = 10,
        stale_min_pnl_pct: float = 0.002,
    ) -> None:
        self.max_holding_bars = max_holding_bars
        self.stale_bars = stale_bars
        self.stale_min_pnl_pct = stale_min_pnl_pct

    def evaluate(
        self,
        bars_held: int,
        current_pnl_pct: float = 0.0,
        is_weekend_approaching: bool = False,
    ) -> TimeExitRule:
        """
        Evaluate whether the trade should be exited based on time rules.

        Args:
            bars_held: Number of bars since entry.
            current_pnl_pct: Current P&L as fraction (e.g. 0.01 = +1%).
            is_weekend_approaching: True if next bar crosses into weekend.
        """
        if bars_held >= self.max_holding_bars:
            return TimeExitRule(
                should_exit=True,
                reason=f"max_holding_period_reached ({bars_held}/{self.max_holding_bars} bars)",
                bars_held=bars_held,
                max_bars=self.max_holding_bars,
            )

        if is_weekend_approaching:
            return TimeExitRule(
                should_exit=True,
                reason="weekend_risk_reduction",
                bars_held=bars_held,
                max_bars=self.max_holding_bars,
            )

        if bars_held >= self.stale_bars and abs(current_pnl_pct) < self.stale_min_pnl_pct:
            return TimeExitRule(
                should_exit=True,
                reason=f"stale_trade (held {bars_held} bars, pnl={current_pnl_pct:+.4f})",
                bars_held=bars_held,
                max_bars=self.max_holding_bars,
            )

        return TimeExitRule(
            should_exit=False,
            reason="within_time_limits",
            bars_held=bars_held,
            max_bars=self.max_holding_bars,
        )


# ---------------------------------------------------------------------------
# R-multiple journal
# ---------------------------------------------------------------------------


@dataclass
class RMultipleEntry:
    """Single trade record with R-multiple tracking."""

    timestamp: str
    stock_name: str
    direction: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_per_unit: float  # |entry - stop| = 1R
    status: str  # "open" | "closed"
    exit_price: float = 0.0
    exit_reason: str = ""
    realized_r: float = 0.0  # (exit - entry) / risk_per_unit [signed]
    bars_held: int = 0
    pnl_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMultipleJournal:
    """
    Tracks R-multiple for every trade.

    R = risk per unit = |entry_price - stop_loss_price|
    Realized R-multiple = (exit_price - entry_price) / R  (for LONG)
                        = (entry_price - exit_price) / R  (for SHORT)
    """

    DEFAULT_PATH = Path("logs/r_multiple_journal.json")

    def __init__(self, path: Path | None = None, max_entries: int = 500) -> None:
        self.path = path or self.DEFAULT_PATH
        self.max_entries = max_entries
        self._entries: list[RMultipleEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            for item in data:
                self._entries.append(RMultipleEntry(**item))
        except Exception:
            self._entries = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = [e.to_dict() for e in self._entries]
            self.path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def open_trade(
        self,
        stock_name: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> RMultipleEntry:
        """Record a new trade entry."""
        risk_per_unit = abs(entry_price - stop_loss_price)
        entry = RMultipleEntry(
            timestamp=datetime.now(UTC).isoformat(),
            stock_name=stock_name,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_per_unit=risk_per_unit,
            status="open",
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        self._save()
        return entry

    def close_trade(
        self,
        stock_name: str,
        exit_price: float,
        exit_reason: str,
        bars_held: int = 0,
    ) -> RMultipleEntry | None:
        """Close the most recent open trade for the given stock."""
        for entry in reversed(self._entries):
            if entry.stock_name == stock_name and entry.status == "open":
                entry.exit_price = exit_price
                entry.exit_reason = exit_reason
                entry.bars_held = bars_held
                entry.status = "closed"

                if entry.risk_per_unit > 0:
                    if entry.direction == "LONG":
                        entry.realized_r = (
                            exit_price - entry.entry_price
                        ) / entry.risk_per_unit
                    else:
                        entry.realized_r = (
                            entry.entry_price - exit_price
                        ) / entry.risk_per_unit
                else:
                    entry.realized_r = 0.0

                if entry.entry_price > 0:
                    if entry.direction == "LONG":
                        entry.pnl_pct = (exit_price - entry.entry_price) / entry.entry_price
                    else:
                        entry.pnl_pct = (entry.entry_price - exit_price) / entry.entry_price
                else:
                    entry.pnl_pct = 0.0

                self._save()
                return entry
        return None

    def get_entries(
        self,
        stock_name: str | None = None,
        status: str | None = None,
    ) -> list[RMultipleEntry]:
        """Return entries, optionally filtered."""
        entries = list(self._entries)
        if stock_name:
            entries = [e for e in entries if e.stock_name.upper() == stock_name.upper()]
        if status:
            entries = [e for e in entries if e.status == status]
        return entries

    def summary(self) -> dict[str, Any]:
        """Aggregate R-multiple statistics for closed trades."""
        closed = [e for e in self._entries if e.status == "closed"]
        if not closed:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "avg_r": 0.0,
                "best_r": 0.0,
                "worst_r": 0.0,
                "win_rate": 0.0,
                "expectancy_r": 0.0,
            }

        r_values = [e.realized_r for e in closed]
        winners = [r for r in r_values if r > 0]
        losers = [r for r in r_values if r <= 0]
        win_rate = len(winners) / len(closed) if closed else 0.0
        avg_win = sum(winners) / len(winners) if winners else 0.0
        avg_loss = sum(losers) / len(losers) if losers else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "total_trades": len(closed),
            "winners": len(winners),
            "losers": len(losers),
            "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
            "best_r": round(max(r_values), 4) if r_values else 0.0,
            "worst_r": round(min(r_values), 4) if r_values else 0.0,
            "win_rate": round(win_rate, 4),
            "expectancy_r": round(expectancy, 4),
        }

    def clear(self) -> None:
        self._entries = []
        self._save()


# ---------------------------------------------------------------------------
# Execution & Exit Plan (composite output)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionExitPlan:
    """Complete exit plan produced by the ExecutionExitManager."""

    # Entry validation
    entry_valid: bool
    entry_direction: str
    entry_confidence: float
    entry_rejection_reasons: list[str] = field(default_factory=list)

    # Stop-loss
    stop_loss_method: str = "fixed"
    stop_loss_price: float = 0.0
    stop_loss_distance_pct: float = 0.0

    # Take-profit
    take_profit_price: float = 0.0
    take_profit_distance_pct: float = 0.0
    trailing_active: bool = False
    trailing_offset_pct: float = 0.0
    partial_targets: list[dict[str, float]] = field(default_factory=list)

    # Time exit
    time_exit_active: bool = False
    max_holding_bars: int = 20

    # R-multiple
    risk_per_unit: float = 0.0
    initial_r_target: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_prompt_section(self) -> str:
        """Format as a text section for injection into the graph state."""
        if not self.entry_valid:
            reasons = ", ".join(self.entry_rejection_reasons)
            return (
                f"### Execution & Exit Plan:\n"
                f"- **Entry REJECTED**: {reasons}\n"
                f"- No trade should be executed.\n"
            )

        lines = [
            "### Execution & Exit Plan:",
            f"- **Direction**: {self.entry_direction} (confidence: {self.entry_confidence:.0%})",
            f"- **Stop-loss**: {self.stop_loss_price:.4f} ({self.stop_loss_method}, "
            f"-{self.stop_loss_distance_pct:.2%} from entry)",
            f"- **Take-profit**: {self.take_profit_price:.4f} "
            f"(+{self.take_profit_distance_pct:.2%} from entry)",
            f"- **Risk per unit (1R)**: {self.risk_per_unit:.4f}",
            f"- **Initial R-target**: {self.initial_r_target:.1f}R",
        ]

        if self.trailing_active:
            lines.append(
                f"- **Trailing stop**: active (offset {self.trailing_offset_pct:.2%})"
            )

        if self.partial_targets:
            targets_str = ", ".join(
                f"{t['r_multiple']}R→{t['exit_pct']:.0%}" for t in self.partial_targets
            )
            lines.append(f"- **Partial exits**: {targets_str}")

        lines.append(
            f"- **Max holding**: {self.max_holding_bars} bars (time-based exit active)"
        )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ExecutionExitManager — orchestrator
# ---------------------------------------------------------------------------


class ExecutionExitManager:
    """
    Orchestrates the full Execution & Exits layer.

    Processes the decision output and produces a complete exit plan
    with stop-loss, take-profit, trailing, time exits, and R-multiple logging.
    """

    def __init__(
        self,
        entry_validator: EntryValidator | None = None,
        stop_loss_calc: StopLossCalculator | None = None,
        take_profit_mgr: TakeProfitManager | None = None,
        time_exit: TimeBasedExit | None = None,
        r_journal: RMultipleJournal | None = None,
    ) -> None:
        self.entry_validator = entry_validator or EntryValidator()
        self.stop_loss_calc = stop_loss_calc or StopLossCalculator()
        self.take_profit_mgr = take_profit_mgr or TakeProfitManager()
        self.time_exit = time_exit or TimeBasedExit()
        self.r_journal = r_journal or RMultipleJournal()

    def process(self, state: dict[str, Any], decision_raw: str) -> ExecutionExitPlan:
        """
        Process a trade decision and produce a complete exit plan.

        This is the main entry point, called as a graph node between
        Decision Maker and the global risk layer.

        Args:
            state: Current graph state dict.
            decision_raw: Raw LLM decision output (JSON string).

        Returns:
            ExecutionExitPlan with all exit parameters.
        """
        # 1. Parse decision
        decision_data = self._parse_decision(decision_raw)

        # 2. Validate entry
        validation = self.entry_validator.validate(decision_data)

        if not validation.is_valid:
            return ExecutionExitPlan(
                entry_valid=False,
                entry_direction=validation.direction,
                entry_confidence=validation.confidence,
                entry_rejection_reasons=validation.rejection_reasons,
            )

        direction = validation.direction
        kline = state.get("kline_data", {})
        closes = kline.get("close", [])
        highs = kline.get("high", [])
        lows = kline.get("low", [])

        # Use last close as entry price
        entry_price = closes[-1] if closes else 100.0

        # 3. Compute stop-loss
        stop_level = self.stop_loss_calc.compute(
            direction=direction,
            entry_price=entry_price,
            highs=highs or None,
            lows=lows or None,
            closes=closes or None,
        )

        # 4. Compute take-profit / trailing
        stop_distance = abs(entry_price - stop_level.stop_price)
        tp_plan = self.take_profit_mgr.compute(
            direction=direction,
            entry_price=entry_price,
            stop_distance=stop_distance,
        )

        # 5. R-multiple tracking
        risk_per_unit = stop_distance
        initial_r = (
            abs(tp_plan.initial_target_price - entry_price) / risk_per_unit
            if risk_per_unit > 0
            else 0.0
        )

        # Log trade in R-journal
        stock_name = str(state.get("stock_name", "UNKNOWN"))
        try:
            self.r_journal.open_trade(
                stock_name=stock_name,
                direction=direction,
                entry_price=entry_price,
                stop_loss_price=stop_level.stop_price,
                take_profit_price=tp_plan.initial_target_price,
            )
        except Exception:
            pass  # best-effort

        return ExecutionExitPlan(
            entry_valid=True,
            entry_direction=direction,
            entry_confidence=validation.confidence,
            # Stop-loss
            stop_loss_method=stop_level.method,
            stop_loss_price=stop_level.stop_price,
            stop_loss_distance_pct=stop_level.distance_pct,
            # Take-profit
            take_profit_price=tp_plan.initial_target_price,
            take_profit_distance_pct=tp_plan.initial_target_pct,
            trailing_active=tp_plan.trailing_active,
            trailing_offset_pct=tp_plan.trailing_offset_pct,
            partial_targets=tp_plan.partial_targets,
            # Time
            time_exit_active=True,
            max_holding_bars=self.time_exit.max_holding_bars,
            # R-multiple
            risk_per_unit=round(risk_per_unit, 4),
            initial_r_target=round(initial_r, 2),
        )

    @staticmethod
    def _parse_decision(raw: str) -> dict[str, Any]:
        """Best-effort parse of LLM decision JSON."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}


# ---------------------------------------------------------------------------
# Graph node factory
# ---------------------------------------------------------------------------


def create_execution_exit_node(
    manager: ExecutionExitManager | None = None,
) -> Any:
    """
    Create a LangGraph node that applies the Execution & Exits layer.

    Reads the decision from state["final_trade_decision"],
    produces an execution plan, and stores it in the state.
    """
    if manager is None:
        manager = ExecutionExitManager()

    def execution_exit_node(state: dict[str, Any]) -> dict[str, Any]:
        decision_raw = state.get("final_trade_decision", "")
        plan = manager.process(state, decision_raw)

        return {
            "execution_exit_plan": plan.to_dict(),
            "execution_exit_summary": plan.format_prompt_section(),
        }

    return execution_exit_node
