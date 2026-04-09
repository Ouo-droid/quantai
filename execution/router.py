"""
execution/router.py
--------------------
Router d'exécution vers Alpaca paper trading.

Flux :
    TradeOrder → RiskEngine.validate() → AlpacaRouter.submit() → Fill

Alpaca paper trading uniquement — jamais l'API réelle.
Vérifie que APCA_BASE_URL contient "paper" au démarrage.

Usage :
    router = AlpacaRouter()
    if router.is_available():
        fill = router.submit(order, symbol="AAPL")
        print(fill)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

APCA_KEY    = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
APCA_URL    = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")


@dataclass
class OrderFill:
    """Résultat d'un ordre soumis à Alpaca."""

    order_id: str
    symbol: str
    direction: str            # "buy" | "sell"
    qty: float
    filled_qty: float
    filled_avg_price: float | None
    status: str               # "new" | "filled" | "partially_filled" | "rejected"
    submitted_at: datetime
    rationale: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status in ("filled", "partially_filled")

    def __str__(self) -> str:
        price = f"@{self.filled_avg_price:.2f}" if self.filled_avg_price else ""
        return (
            f"OrderFill({self.direction.upper()} {self.filled_qty:.4f} {self.symbol} "
            f"{price} status={self.status})"
        )


class AlpacaRouter:
    """
    Soumet les TradeOrder à Alpaca paper trading.
    Non bloquant : si Alpaca indisponible → retourne None.
    """

    def __init__(self) -> None:
        self._safe_url = APCA_URL

        # Garde de sécurité — refuser si URL ne contient pas "paper"
        if "paper" not in APCA_URL.lower():
            raise ValueError(
                f"APCA_BASE_URL doit contenir 'paper' pour du paper trading. "
                f"URL actuelle : {APCA_URL}"
            )

    def is_available(self) -> bool:
        """Vérifie que les clés Alpaca sont présentes et le serveur joignable."""
        if not APCA_KEY or not APCA_SECRET:
            logger.info("Alpaca : clés API manquantes (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
            return False
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
            account: Any = client.get_account()
            cash = float(account.cash) if hasattr(account, "cash") else float(account.get("cash", 0))
            equity = float(account.equity) if hasattr(account, "equity") else float(account.get("equity", 0))
            logger.info(f"Alpaca paper account : cash=${cash:,.0f} equity=${equity:,.0f}")
            return True
        except Exception as e:
            logger.warning(f"Alpaca non disponible : {e}")
            return False

    def get_account(self) -> dict | None:
        """Retourne les infos du compte paper."""
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
            acc: Any = client.get_account()

            def _f(val: Any) -> float:
                return float(val) if val is not None else 0.0

            if isinstance(acc, dict):
                cash = _f(acc.get("cash"))
                equity = _f(acc.get("equity"))
                bp = _f(acc.get("buying_power"))
                pv = _f(acc.get("portfolio_value"))
                last_eq = _f(acc.get("last_equity"))
                status = acc.get("status", "unknown")
            else:
                cash = _f(acc.cash)
                equity = _f(acc.equity)
                bp = _f(acc.buying_power)
                pv = _f(acc.portfolio_value)
                last_eq = _f(acc.last_equity)
                status = acc.status

            return {
                "cash":            cash,
                "equity":          equity,
                "buying_power":    bp,
                "portfolio_value": pv,
                "pnl_today":       equity - last_eq,
                "status":          status,
            }
        except Exception as e:
            logger.warning(f"get_account() failed : {e}")
            return None

    def get_positions(self) -> list[dict]:
        """Retourne les positions ouvertes."""
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
            positions = client.get_all_positions()

            def _f(val: Any) -> float:
                return float(val) if val is not None else 0.0

            results = []
            for p in positions:
                if isinstance(p, str):
                    continue
                results.append({
                    "symbol":         p.symbol,
                    "qty":            _f(p.qty),
                    "side":           p.side.value if hasattr(p.side, "value") else str(p.side),
                    "avg_entry":      _f(p.avg_entry_price),
                    "market_val":     _f(p.market_value),
                    "unrealized_pnl": _f(p.unrealized_pl),
                    "unrealized_pct": _f(p.unrealized_plpc) * 100,
                })
            return results
        except Exception as e:
            logger.warning(f"get_positions() failed : {e}")
            return []

    def get_orders(self, limit: int = 20) -> list[dict]:
        """Retourne les derniers ordres."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import GetOrdersRequest

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
            orders = client.get_orders(filter=GetOrdersRequest(limit=limit))
            results = []
            for o in orders:
                if isinstance(o, str):
                    continue
                results.append({
                    "id":         str(o.id),
                    "symbol":     o.symbol,
                    "side":       o.side.value if (hasattr(o.side, "value") and o.side) else str(o.side),
                    "qty":        float(o.qty or 0),
                    "filled_qty": float(o.filled_qty or 0),
                    "status":     o.status.value if hasattr(o.status, "value") else str(o.status),
                    "created_at": str(o.created_at),
                })
            return results
        except Exception as e:
            logger.warning(f"get_orders() failed : {e}")
            return []

    def submit(
        self,
        order: Any,
        symbol: str,
        account_value: float = 100_000.0,
    ) -> OrderFill | None:
        """
        Soumet un TradeOrder à Alpaca paper trading.

        Args:
            order         : TradeOrder (direction, size_pct, stop_loss, take_profit)
            symbol        : ticker (ex: "AAPL")
            account_value : valeur du compte pour calculer la taille en $ (défaut 100k)

        Returns:
            OrderFill si soumis · None si FLAT, indisponible ou erreur
        """
        if order.direction == "FLAT":
            logger.info(f"{symbol} : ordre FLAT → pas de soumission Alpaca")
            return None

        if not self.is_available():
            return None

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)

            notional = round(account_value * order.size_pct, 2)
            if notional < 1.0:
                logger.warning(f"{symbol} : notionnel trop faible ({notional:.2f}$)")
                return None

            side = OrderSide.BUY if order.direction == "LONG" else OrderSide.SELL

            market_order = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

            response = client.submit_order(market_order)

            resp: Any = response
            fill = OrderFill(
                order_id=str(resp.id) if hasattr(resp, "id") else str(resp.get("id")),
                symbol=symbol,
                direction=order.direction.lower(),
                qty=float(resp.qty or 0) if hasattr(resp, "qty") else float(resp.get("qty", 0)),
                filled_qty=float(resp.filled_qty or 0) if hasattr(resp, "filled_qty") else float(resp.get("filled_qty", 0)),
                filled_avg_price=(
                    float(resp.filled_avg_price) if (hasattr(resp, "filled_avg_price") and resp.filled_avg_price) else (float(resp.get("filled_avg_price")) if resp.get("filled_avg_price") else None)
                ),
                status=resp.status.value if hasattr(resp.status, "value") else str(resp.status),
                submitted_at=(resp.created_at if hasattr(resp, "created_at") else resp.get("created_at")) or datetime.now(),
                rationale=getattr(order, "rationale", ""),
                raw={"id": str(resp.id) if hasattr(resp, "id") else str(resp.get("id")), "status": resp.status.value if hasattr(resp.status, "value") else str(resp.status)},
            )

            logger.info(f"{symbol} → {fill}")
            return fill

        except Exception as e:
            logger.error(f"{symbol} : Alpaca submit failed — {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """Ferme une position existante."""
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
            client.close_position(symbol)
            logger.info(f"{symbol} : position fermée")
            return True
        except Exception as e:
            logger.warning(f"{symbol} : close_position failed — {e}")
            return False

    def close_all_positions(self) -> bool:
        """Ferme toutes les positions (fin de session)."""
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
            client.close_all_positions(cancel_orders=True)
            logger.info("Toutes les positions fermées")
            return True
        except Exception as e:
            logger.warning(f"close_all_positions failed — {e}")
            return False
