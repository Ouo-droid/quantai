"""
execution/risk.py
-----------------
Risk Engine — filtre synchrone entre DecisionAgent et l'exécution réelle.

Aucun ordre ne part sans passer par RiskEngine.validate().

Flux :
    DecisionAgent → TradeOrder → RiskEngine.validate() → TradeOrder modifié ou None → Execution

Usage :
    portfolio = Portfolio(cash=100_000)
    engine = RiskEngine(portfolio)
    safe_order = engine.validate(order, recent_returns=daily_returns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from execution.decision_agent import TradeOrder


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@dataclass
class Portfolio:
    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    # symbol → valeur en USD (positif = long, négatif = short)
    equity_history: list[float] = field(default_factory=list)
    # historique de la valeur totale (pour calcul drawdown)

    @property
    def total_equity(self) -> float:
        return self.cash + sum(abs(v) for v in self.positions.values())

    def position_pct(self, symbol: str) -> float:
        """Poids d'une position en % du portefeuille total."""
        equity = self.total_equity
        if equity == 0:
            return 0.0
        return abs(self.positions.get(symbol, 0.0)) / equity

    def max_drawdown(self) -> float:
        """Max drawdown depuis le pic — retourne valeur négative (ex: -0.12 pour -12%)."""
        if len(self.equity_history) < 2:
            return 0.0
        peak = self.equity_history[0]
        max_dd = 0.0
        for value in self.equity_history[1:]:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (value - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        return max_dd

    def var_95(self, returns: list[float]) -> float:
        """
        VaR historique 95% sur les rendements journaliers fournis.

        Retourne la perte au 5e percentile comme valeur positive
        (ex: 0.025 signifie une VaR de 2.5%).
        """
        if not returns:
            return 0.0
        sorted_returns = sorted(returns)
        index = max(0, int(len(sorted_returns) * 0.05) - 1)
        var = -sorted_returns[index]
        return max(0.0, var)


# ---------------------------------------------------------------------------
# RiskLimits
# ---------------------------------------------------------------------------


@dataclass
class RiskLimits:
    max_position_pct: float = 0.05  # 5% max par position
    max_sector_pct: float = 0.25  # 25% max par secteur
    daily_var_95: float = 0.02  # VaR 95% < 2% du portefeuille
    max_drawdown: float = 0.10  # drawdown max 10% (valeur positive)
    min_confidence: float = 0.60  # confidence minimale
    max_leverage: float = 1.0  # pas de levier


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------


class RiskEngine:
    def __init__(
        self,
        portfolio: Portfolio,
        limits: RiskLimits | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.limits = limits or RiskLimits()
        self._log: list[dict] = []

    def validate(
        self,
        order: TradeOrder,
        recent_returns: list[float] | None = None,
    ) -> TradeOrder | None:
        """
        Valide et ajuste un ordre selon les limites de risque.

        Returns:
            TradeOrder modifié (size_pct ajusté) si l'ordre passe.
            None si l'ordre est bloqué.
        """
        # FLAT orders always pass — they represent inaction
        if order.direction == "FLAT":
            self._record(order.symbol, "pass", "direction=FLAT → always allowed", None, None)
            return order

        # 1. Confidence check
        if not self._check_confidence(order):
            self._record(
                order.symbol,
                "blocked",
                "confidence_too_low",
                order.confidence,
                self.limits.min_confidence,
            )
            logger.warning(
                f"[RiskEngine] {order.symbol} blocked: confidence {order.confidence:.2f} "
                f"< limit {self.limits.min_confidence:.2f}"
            )
            return None

        # 2. Drawdown check
        if not self._check_drawdown():
            current_dd = self.portfolio.max_drawdown()
            self._record(
                order.symbol,
                "blocked",
                "drawdown_exceeded",
                current_dd,
                -self.limits.max_drawdown,
            )
            logger.warning(
                f"[RiskEngine] {order.symbol} blocked: drawdown {current_dd:.1%} "
                f"> limit -{self.limits.max_drawdown:.1%}"
            )
            return None

        # 3. VaR check
        if recent_returns is not None and len(recent_returns) >= 20:
            if not self._check_var(recent_returns):
                current_var = self.portfolio.var_95(recent_returns)
                self._record(
                    order.symbol,
                    "blocked",
                    "var_exceeded",
                    current_var,
                    self.limits.daily_var_95,
                )
                logger.warning(
                    f"[RiskEngine] {order.symbol} blocked: VaR {current_var:.1%} > limit {self.limits.daily_var_95:.1%}"
                )
                return None

        # 4. Position size check (adjusts, does not block)
        order = self._check_position_size(order)

        self._record(
            order.symbol,
            "pass",
            "all_checks_passed",
            order.size_pct,
            self.limits.max_position_pct,
        )
        logger.info(f"[RiskEngine] {order.symbol} → {order.direction} size={order.size_pct:.1%} approved")
        return order

    def _check_position_size(self, order: TradeOrder) -> TradeOrder:
        """Réduit size_pct si dépasse max_position_pct (incluant l'exposition actuelle)."""
        current_pct = self.portfolio.position_pct(order.symbol)
        total_pct = current_pct + order.size_pct

        if total_pct > self.limits.max_position_pct:
            original = order.size_pct
            # On ne peut ajouter que ce qui manque pour atteindre la limite
            allowed_add = max(0.0, self.limits.max_position_pct - current_pct)
            order.size_pct = allowed_add

            logger.info(
                f"[RiskEngine] {order.symbol} position size adjusted {original:.1%} → {order.size_pct:.1%} "
                f"(current={current_pct:.1%}, limit={self.limits.max_position_pct:.1%})"
            )
            self._record(
                order.symbol,
                "adjusted",
                "position_size_reduced",
                original,
                allowed_add,
            )
        return order

    def _check_drawdown(self) -> bool:
        """Retourne True si drawdown acceptable, False si dépassé."""
        current_dd = self.portfolio.max_drawdown()
        # max_drawdown() returns negative value; limits.max_drawdown is positive
        return current_dd > -self.limits.max_drawdown

    def _check_var(self, recent_returns: list[float]) -> bool:
        """Retourne True si VaR acceptable, False si dépassée."""
        var = self.portfolio.var_95(recent_returns)
        return var <= self.limits.daily_var_95

    def _check_confidence(self, order: TradeOrder) -> bool:
        """Double check confidence minimum."""
        return order.confidence >= self.limits.min_confidence

    def _record(
        self,
        symbol: str,
        outcome: str,
        rule: str,
        observed: float | None,
        limit: float | None,
    ) -> None:
        self._log.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "outcome": outcome,
                "rule": rule,
                "observed": observed,
                "limit": limit,
            }
        )

    def audit_log(self) -> list[dict]:
        """Retourne l'historique de toutes les décisions de risque."""
        return list(self._log)
