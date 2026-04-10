"""
Regime Risk Calibrator — adjusts risk parameters based on detected market regime.

Detects the current market regime (BULL, BEAR, HIGH_VOL, LOW_VOL, CRISIS) from
recent price data and returns calibrated risk parameters that the decision agent
can use to adjust position sizing, stop-losses, and confidence thresholds.

Usage:
    calibrator = RegimeRiskCalibrator()
    regime = calibrator.detect_regime(recent_returns)
    params = calibrator.get_risk_params(regime)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Market Regimes
# ---------------------------------------------------------------------------


class MarketRegime(StrEnum):
    """Identifiable market regimes."""

    BULL = "BULL"
    BEAR = "BEAR"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    CRISIS = "CRISIS"


# ---------------------------------------------------------------------------
# Calibrated risk parameters per regime
# ---------------------------------------------------------------------------


@dataclass
class RegimeRiskParams:
    """Risk parameters calibrated for a specific market regime."""

    regime: MarketRegime
    max_position_pct: float  # max % of portfolio per position
    stop_loss_pct: float  # stop-loss distance as % of price
    take_profit_pct: float  # take-profit distance as % of price
    min_confidence: float  # minimum confidence to enter a trade
    risk_reward_min: float  # minimum acceptable risk-reward ratio
    risk_reward_max: float  # maximum risk-reward ratio to suggest
    size_scalar: float  # multiplier applied to base position size (1.0 = normal)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "max_position_pct": self.max_position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_confidence": self.min_confidence,
            "risk_reward_min": self.risk_reward_min,
            "risk_reward_max": self.risk_reward_max,
            "size_scalar": self.size_scalar,
            "description": self.description,
        }


# Default regime parameters
_REGIME_PARAMS: dict[MarketRegime, RegimeRiskParams] = {
    MarketRegime.BULL: RegimeRiskParams(
        regime=MarketRegime.BULL,
        max_position_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        min_confidence=0.60,
        risk_reward_min=1.2,
        risk_reward_max=1.8,
        size_scalar=1.0,
        description="Normal bull market — standard risk parameters",
    ),
    MarketRegime.BEAR: RegimeRiskParams(
        regime=MarketRegime.BEAR,
        max_position_pct=0.03,
        stop_loss_pct=0.015,
        take_profit_pct=0.04,
        min_confidence=0.70,
        risk_reward_min=1.5,
        risk_reward_max=2.0,
        size_scalar=0.6,
        description="Bear market — tighter stops, higher confidence required, smaller positions",
    ),
    MarketRegime.HIGH_VOL: RegimeRiskParams(
        regime=MarketRegime.HIGH_VOL,
        max_position_pct=0.03,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        min_confidence=0.75,
        risk_reward_min=1.5,
        risk_reward_max=2.5,
        size_scalar=0.5,
        description="High volatility — wider stops, much smaller positions, high confidence bar",
    ),
    MarketRegime.LOW_VOL: RegimeRiskParams(
        regime=MarketRegime.LOW_VOL,
        max_position_pct=0.05,
        stop_loss_pct=0.01,
        take_profit_pct=0.03,
        min_confidence=0.55,
        risk_reward_min=1.2,
        risk_reward_max=1.5,
        size_scalar=1.2,
        description="Low volatility — tighter stops, slightly larger positions allowed",
    ),
    MarketRegime.CRISIS: RegimeRiskParams(
        regime=MarketRegime.CRISIS,
        max_position_pct=0.01,
        stop_loss_pct=0.01,
        take_profit_pct=0.03,
        min_confidence=0.85,
        risk_reward_min=2.0,
        risk_reward_max=3.0,
        size_scalar=0.25,
        description="Crisis regime — minimal exposure, very high confidence required",
    ),
}


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


@dataclass
class RegimeDetectionResult:
    """Result of regime detection with diagnostic info."""

    regime: MarketRegime
    annualised_vol: float
    trend_return: float  # cumulative return over the lookback
    max_drawdown: float  # max drawdown over the lookback (negative)
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "annualised_vol": round(self.annualised_vol, 4),
            "trend_return": round(self.trend_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "diagnostics": self.diagnostics,
        }


class RegimeRiskCalibrator:
    """
    Detects the current market regime from recent returns and provides
    calibrated risk parameters.

    Regime detection heuristic (based on annualised volatility, trend, and drawdown):

    1. CRISIS:  max_drawdown < -15% OR annualised_vol > 40%
    2. HIGH_VOL: annualised_vol > 25%
    3. BEAR:    trend_return < -5%
    4. LOW_VOL: annualised_vol < 10%
    5. BULL:    default (positive trend, normal vol)
    """

    # Thresholds (can be overridden)
    CRISIS_DRAWDOWN = -0.15
    CRISIS_VOL = 0.40
    HIGH_VOL_THRESHOLD = 0.25
    BEAR_RETURN_THRESHOLD = -0.05
    LOW_VOL_THRESHOLD = 0.10

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        custom_params: dict[MarketRegime, RegimeRiskParams] | None = None,
    ) -> None:
        self._params = dict(_REGIME_PARAMS)
        if custom_params:
            self._params.update(custom_params)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        recent_returns: list[float],
        lookback: int | None = None,
    ) -> RegimeDetectionResult:
        """
        Detect the current market regime from a list of daily returns.

        Args:
            recent_returns: Daily simple returns (e.g. [0.01, -0.005, ...]).
            lookback: Number of recent days to use. Defaults to all.

        Returns:
            RegimeDetectionResult with the detected regime and diagnostics.
        """
        if lookback is not None:
            returns = recent_returns[-lookback:]
        else:
            returns = list(recent_returns)

        if len(returns) < 5:
            # Too little data — assume bull (safest default)
            return RegimeDetectionResult(
                regime=MarketRegime.BULL,
                annualised_vol=0.0,
                trend_return=0.0,
                max_drawdown=0.0,
                diagnostics={"reason": "insufficient_data", "n_returns": len(returns)},
            )

        # Compute statistics
        ann_vol = self._annualised_vol(returns)
        trend_ret = self._cumulative_return(returns)
        max_dd = self._max_drawdown(returns)

        diagnostics: dict[str, Any] = {
            "n_returns": len(returns),
            "daily_vol": round(self._daily_vol(returns), 6),
            "ann_vol": round(ann_vol, 4),
            "trend_return": round(trend_ret, 4),
            "max_drawdown": round(max_dd, 4),
        }

        # Priority-ordered regime detection
        if max_dd < self.CRISIS_DRAWDOWN or ann_vol > self.CRISIS_VOL:
            regime = MarketRegime.CRISIS
            diagnostics["trigger"] = (
                "drawdown" if max_dd < self.CRISIS_DRAWDOWN else "extreme_vol"
            )
        elif ann_vol > self.HIGH_VOL_THRESHOLD:
            regime = MarketRegime.HIGH_VOL
            diagnostics["trigger"] = "high_vol"
        elif trend_ret < self.BEAR_RETURN_THRESHOLD:
            regime = MarketRegime.BEAR
            diagnostics["trigger"] = "negative_trend"
        elif ann_vol < self.LOW_VOL_THRESHOLD:
            regime = MarketRegime.LOW_VOL
            diagnostics["trigger"] = "low_vol"
        else:
            regime = MarketRegime.BULL
            diagnostics["trigger"] = "default"

        return RegimeDetectionResult(
            regime=regime,
            annualised_vol=ann_vol,
            trend_return=trend_ret,
            max_drawdown=max_dd,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Risk parameters
    # ------------------------------------------------------------------

    def get_risk_params(self, regime: MarketRegime) -> RegimeRiskParams:
        """Return calibrated risk parameters for the given regime."""
        return self._params[regime]

    def calibrate(
        self, recent_returns: list[float], lookback: int | None = None
    ) -> tuple[RegimeDetectionResult, RegimeRiskParams]:
        """
        Convenience method: detect regime and return calibrated params in one call.

        Returns:
            (detection_result, risk_params)
        """
        result = self.detect_regime(recent_returns, lookback)
        params = self.get_risk_params(result.regime)
        return result, params

    def format_regime_prompt_section(
        self,
        result: RegimeDetectionResult,
        params: RegimeRiskParams,
    ) -> str:
        """
        Format a text section suitable for injection into the LLM decision prompt.

        This tells the decision agent about the current market regime
        and the recommended risk parameters.
        """
        return (
            f"### 5. Market Regime Risk Calibration:\n"
            f"- **Detected regime**: {result.regime.value} — {params.description}\n"
            f"- Annualised volatility: {result.annualised_vol:.1%}\n"
            f"- Lookback trend return: {result.trend_return:.1%}\n"
            f"- Max drawdown (lookback): {result.max_drawdown:.1%}\n"
            f"- **Recommended parameters**:\n"
            f"  - Max position size: {params.max_position_pct:.1%}\n"
            f"  - Stop-loss: {params.stop_loss_pct:.1%}\n"
            f"  - Take-profit: {params.take_profit_pct:.1%}\n"
            f"  - Minimum confidence: {params.min_confidence:.0%}\n"
            f"  - Risk-reward ratio range: {params.risk_reward_min}–{params.risk_reward_max}\n"
            f"  - Position size scalar: {params.size_scalar}x\n"
            f"- Adjust your decision confidence thresholds and risk-reward ratio "
            f"according to these regime-calibrated parameters.\n"
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_vol(returns: list[float]) -> float:
        """Standard deviation of daily returns."""
        n = len(returns)
        if n < 2:
            return 0.0
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        return variance**0.5

    def _annualised_vol(self, returns: list[float]) -> float:
        """Annualised volatility (daily vol × √252)."""
        return self._daily_vol(returns) * (self.TRADING_DAYS_PER_YEAR**0.5)

    @staticmethod
    def _cumulative_return(returns: list[float]) -> float:
        """Cumulative return from a list of simple daily returns."""
        cum = 1.0
        for r in returns:
            cum *= 1.0 + r
        return cum - 1.0

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        """
        Max drawdown from a list of daily returns.
        Returns a negative number (e.g. -0.12 for -12%).
        """
        cum = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            cum *= 1.0 + r
            if cum > peak:
                peak = cum
            dd = (cum - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd
