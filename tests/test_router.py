"""
tests/test_router.py
---------------------
Tests du AlpacaRouter — sans clés Alpaca réelles (mocks uniquement).

Lance : .venv/bin/python -m pytest tests/test_router.py -v
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from execution.router import AlpacaRouter, OrderFill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_order(direction: str = "LONG", confidence: float = 0.75, size_pct: float = 0.04):
    """Crée un objet order minimal compatible avec AlpacaRouter.submit()."""
    order = MagicMock()
    order.direction = direction
    order.confidence = confidence
    order.size_pct = size_pct
    order.rationale = "test rationale"
    return order


def make_fill(**kwargs) -> OrderFill:
    defaults = dict(
        order_id="abc-123",
        symbol="AAPL",
        direction="buy",
        qty=1.0,
        filled_qty=1.0,
        filled_avg_price=150.0,
        status="filled",
        submitted_at=datetime.now(),
    )
    defaults.update(kwargs)
    return OrderFill(**defaults)


# ---------------------------------------------------------------------------
# AlpacaRouter — instanciation et garde de sécurité
# ---------------------------------------------------------------------------


class TestAlpacaRouterInit:
    def test_router_raises_if_not_paper_url(self):
        """AlpacaRouter lève ValueError si APCA_BASE_URL ne contient pas 'paper'."""
        with patch("execution.router.APCA_URL", "https://api.alpaca.markets"):
            with pytest.raises(ValueError, match="paper"):
                AlpacaRouter()

    def test_router_ok_with_paper_url(self):
        """AlpacaRouter s'instancie sans erreur avec une URL paper."""
        with patch("execution.router.APCA_URL", "https://paper-api.alpaca.markets"):
            router = AlpacaRouter()
            assert router is not None

    def test_router_ok_with_default_url(self):
        """L'URL par défaut contient 'paper' — pas d'exception."""
        router = AlpacaRouter()
        assert "paper" in router._safe_url.lower()


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_is_available_false_without_keys(self):
        """is_available() retourne False si clés absentes."""
        with patch("execution.router.APCA_KEY", ""):
            with patch("execution.router.APCA_SECRET", ""):
                router = AlpacaRouter()
                assert router.is_available() is False

    def test_is_available_false_on_api_error(self):
        """is_available() retourne False si l'API est injoignable."""
        with patch("execution.router.APCA_KEY", "fake-key"):
            with patch("execution.router.APCA_SECRET", "fake-secret"):
                router = AlpacaRouter()
                with patch("alpaca.trading.client.TradingClient") as mock_cls:
                    mock_cls.return_value.get_account.side_effect = Exception("Connection refused")
                    assert router.is_available() is False

    def test_is_available_true_with_mock(self):
        """is_available() retourne True quand l'API répond."""
        with patch("execution.router.APCA_KEY", "fake-key"):
            with patch("execution.router.APCA_SECRET", "fake-secret"):
                router = AlpacaRouter()
                mock_account = MagicMock()
                mock_account.cash = "100000.00"
                mock_account.equity = "100000.00"
                with patch("alpaca.trading.client.TradingClient") as mock_cls:
                    mock_cls.return_value.get_account.return_value = mock_account
                    assert router.is_available() is True


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_returns_none_for_flat_order(self):
        """submit() retourne None si direction=FLAT."""
        router = AlpacaRouter()
        order = make_order(direction="FLAT", size_pct=0.0)
        result = router.submit(order, symbol="AAPL")
        assert result is None

    def test_submit_returns_none_when_unavailable(self):
        """submit() retourne None si is_available()=False."""
        router = AlpacaRouter()
        with patch.object(router, "is_available", return_value=False):
            order = make_order(direction="LONG")
            result = router.submit(order, symbol="AAPL")
            assert result is None

    def test_submit_never_raises(self):
        """submit() ne lève jamais d'exception, même avec des inputs pathologiques."""
        router = AlpacaRouter()
        try:
            router.submit(make_order(direction="LONG"), symbol="AAPL")
            router.submit(make_order(direction="SHORT", size_pct=0.0), symbol="")
            order_bad = MagicMock()
            order_bad.direction = "LONG"
            order_bad.size_pct = "not_a_float"
            router.submit(order_bad, symbol="X")
        except Exception as e:
            pytest.fail(f"submit() a levé une exception : {e}")

    def test_notional_too_small_returns_none(self):
        """submit() retourne None si notionnel < 1$ (size_pct trop faible)."""
        router = AlpacaRouter()
        order = make_order(direction="LONG", size_pct=0.000005)
        with patch.object(router, "is_available", return_value=True):
            result = router.submit(order, symbol="AAPL", account_value=100.0)
            assert result is None

    def test_submit_with_mock_alpaca_long(self):
        """submit() avec mock alpaca-py → OrderFill créé correctement pour LONG."""
        router = AlpacaRouter()
        order = make_order(direction="LONG", size_pct=0.04)

        mock_response = MagicMock()
        mock_response.id = "order-xyz-001"
        mock_response.qty = "10.0"
        mock_response.filled_qty = "10.0"
        mock_response.filled_avg_price = "150.25"
        mock_response.status = MagicMock()
        mock_response.status.value = "filled"
        mock_response.created_at = datetime.now()

        with patch.object(router, "is_available", return_value=True):
            with patch("alpaca.trading.client.TradingClient") as mock_cls:
                mock_cls.return_value.submit_order.return_value = mock_response
                fill = router.submit(order, symbol="AAPL", account_value=100_000.0)

        assert fill is not None
        assert isinstance(fill, OrderFill)
        assert fill.symbol == "AAPL"
        assert fill.direction == "long"
        assert fill.status == "filled"
        assert fill.order_id == "order-xyz-001"

    def test_submit_with_mock_alpaca_short(self):
        """submit() soumet un ordre SELL pour direction=SHORT."""
        from alpaca.trading.enums import OrderSide

        router = AlpacaRouter()
        order = make_order(direction="SHORT", size_pct=0.03)

        mock_response = MagicMock()
        mock_response.id = "order-xyz-002"
        mock_response.qty = "5.0"
        mock_response.filled_qty = "0.0"
        mock_response.filled_avg_price = None
        mock_response.status = MagicMock()
        mock_response.status.value = "new"
        mock_response.created_at = datetime.now()

        submitted_request = {}

        def capture_order(req):
            submitted_request["side"] = req.side
            return mock_response

        with patch.object(router, "is_available", return_value=True):
            with patch("alpaca.trading.client.TradingClient") as mock_cls:
                mock_cls.return_value.submit_order.side_effect = capture_order
                fill = router.submit(order, symbol="MSFT", account_value=100_000.0)

        assert fill is not None
        assert fill.direction == "short"
        assert submitted_request["side"] == OrderSide.SELL

    def test_submit_api_error_returns_none(self):
        """submit() retourne None (pas d'exception) si l'API Alpaca plante."""
        router = AlpacaRouter()
        order = make_order(direction="LONG", size_pct=0.04)

        with patch.object(router, "is_available", return_value=True):
            with patch("alpaca.trading.client.TradingClient") as mock_cls:
                mock_cls.return_value.submit_order.side_effect = Exception("API 503")
                result = router.submit(order, symbol="AAPL", account_value=100_000.0)

        assert result is None

    def test_submit_notional_calculation(self):
        """Le notionnel envoyé à Alpaca = account_value × size_pct."""
        router = AlpacaRouter()
        order = make_order(direction="LONG", size_pct=0.05)

        mock_response = MagicMock()
        mock_response.id = "order-notional"
        mock_response.qty = "0.0"
        mock_response.filled_qty = "0.0"
        mock_response.filled_avg_price = None
        mock_response.status = MagicMock()
        mock_response.status.value = "new"
        mock_response.created_at = datetime.now()

        captured = {}

        def capture(req):
            captured["notional"] = req.notional
            return mock_response

        with patch.object(router, "is_available", return_value=True):
            with patch("alpaca.trading.client.TradingClient") as mock_cls:
                mock_cls.return_value.submit_order.side_effect = capture
                router.submit(order, symbol="AAPL", account_value=200_000.0)

        assert captured.get("notional") == pytest.approx(10_000.0, abs=0.01)


# ---------------------------------------------------------------------------
# OrderFill
# ---------------------------------------------------------------------------


class TestOrderFill:
    def test_order_fill_is_filled_true(self):
        """OrderFill.is_filled True si status 'filled'."""
        fill = make_fill(status="filled")
        assert fill.is_filled is True

    def test_order_fill_is_filled_partially(self):
        """OrderFill.is_filled True si status 'partially_filled'."""
        fill = make_fill(status="partially_filled")
        assert fill.is_filled is True

    def test_order_fill_is_filled_false_new(self):
        """OrderFill.is_filled False si status 'new'."""
        fill = make_fill(status="new")
        assert fill.is_filled is False

    def test_order_fill_is_filled_false_rejected(self):
        """OrderFill.is_filled False si status 'rejected'."""
        fill = make_fill(status="rejected")
        assert fill.is_filled is False

    def test_order_fill_str_with_price(self):
        """__str__ inclut le prix si filled_avg_price est défini."""
        fill = make_fill(direction="buy", filled_qty=5.0, symbol="AAPL",
                        filled_avg_price=200.50, status="filled")
        s = str(fill)
        assert "BUY" in s
        assert "200.50" in s
        assert "AAPL" in s

    def test_order_fill_str_no_price(self):
        """__str__ fonctionne sans filled_avg_price."""
        fill = make_fill(direction="sell", filled_avg_price=None, status="new")
        s = str(fill)
        assert "SELL" in s
        assert "new" in s


# ---------------------------------------------------------------------------
# get_account() / get_positions()
# ---------------------------------------------------------------------------


class TestGetAccountPositions:
    def test_get_account_returns_none_when_unavailable(self):
        """get_account() retourne None si l'API plante."""
        with patch("execution.router.APCA_KEY", "fake"):
            with patch("execution.router.APCA_SECRET", "fake"):
                router = AlpacaRouter()
                with patch("alpaca.trading.client.TradingClient") as mock_cls:
                    mock_cls.return_value.get_account.side_effect = Exception("error")
                    assert router.get_account() is None

    def test_get_positions_returns_empty_when_unavailable(self):
        """get_positions() retourne [] si l'API plante."""
        with patch("execution.router.APCA_KEY", "fake"):
            with patch("execution.router.APCA_SECRET", "fake"):
                router = AlpacaRouter()
                with patch("alpaca.trading.client.TradingClient") as mock_cls:
                    mock_cls.return_value.get_all_positions.side_effect = Exception("error")
                    assert router.get_positions() == []

    def test_get_account_returns_dict_with_mock(self):
        """get_account() retourne un dict bien formé si l'API répond."""
        with patch("execution.router.APCA_KEY", "fake"):
            with patch("execution.router.APCA_SECRET", "fake"):
                router = AlpacaRouter()
                mock_acc = MagicMock()
                mock_acc.cash = "50000.00"
                mock_acc.equity = "52000.00"
                mock_acc.buying_power = "100000.00"
                mock_acc.portfolio_value = "52000.00"
                mock_acc.last_equity = "51000.00"
                mock_acc.status = "ACTIVE"
                with patch("alpaca.trading.client.TradingClient") as mock_cls:
                    mock_cls.return_value.get_account.return_value = mock_acc
                    account = router.get_account()

        assert account is not None
        assert account["cash"] == pytest.approx(50_000.0)
        assert account["equity"] == pytest.approx(52_000.0)
        assert account["pnl_today"] == pytest.approx(1_000.0)

    def test_get_positions_parses_positions(self):
        """get_positions() retourne une liste de dicts bien formés."""
        with patch("execution.router.APCA_KEY", "fake"):
            with patch("execution.router.APCA_SECRET", "fake"):
                router = AlpacaRouter()
                mock_pos = MagicMock()
                mock_pos.symbol = "AAPL"
                mock_pos.qty = "10.0"
                mock_pos.side = MagicMock()
                mock_pos.side.value = "long"
                mock_pos.avg_entry_price = "175.00"
                mock_pos.market_value = "1800.00"
                mock_pos.unrealized_pl = "25.00"
                mock_pos.unrealized_plpc = "0.014"
                with patch("alpaca.trading.client.TradingClient") as mock_cls:
                    mock_cls.return_value.get_all_positions.return_value = [mock_pos]
                    positions = router.get_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == pytest.approx(10.0)
        assert positions[0]["unrealized_pct"] == pytest.approx(1.4)
