"""
tests/test_mirofish_client.py
------------------------------
Tests du MiroFishClient — sans serveur MiroFish (mocks uniquement).

Lance : uv run pytest tests/test_mirofish_client.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from simulation.mirofish_client import MiroFishClient, SimulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNewsItem:
    title: str
    text: str = ""
    date: str = ""
    source: str = "test"


def make_client(**kwargs) -> MiroFishClient:
    return MiroFishClient(base_url="http://fake-mirofish:5001", **kwargs)


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_false_when_server_down(self):
        client = make_client()
        # Aucun serveur sur cette adresse — doit retourner False
        assert client.health() is False

    def test_health_true_when_server_responds(self):
        client = make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            assert client.health() is True

    def test_health_false_on_500(self):
        client = make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("httpx.get", return_value=mock_resp):
            assert client.health() is False


# ---------------------------------------------------------------------------
# simulate() — comportement sans serveur
# ---------------------------------------------------------------------------


class TestSimulateWithoutServer:
    def test_simulate_returns_none_when_server_down(self):
        client = make_client()
        seed = [{"title": "Fed raises rates", "content": "...", "date": "", "source": "test"}]
        result = client.simulate(seed, symbol="AAPL")
        assert result.sentiment_index is None

    def test_simulate_never_raises(self):
        client = make_client()
        # Même avec des arguments pathologiques, pas d'exception
        try:
            client.simulate([], symbol="")
            client.simulate([{"bad": "data"}], symbol="X")
        except Exception as e:
            pytest.fail(f"simulate() a levé une exception : {e}")

    def test_simulate_empty_seed_returns_none(self):
        client = make_client()
        result = client.simulate([], symbol="TEST")
        assert result.sentiment_index is None

    def test_simulate_returns_simulation_result_type(self):
        client = make_client()
        result = client.simulate([], symbol="TEST")
        assert isinstance(result, SimulationResult)


# ---------------------------------------------------------------------------
# news_to_seed()
# ---------------------------------------------------------------------------


class TestNewsToSeed:
    def test_news_to_seed_filters_short_titles(self):
        news = [
            FakeNewsItem(title="OK"),           # < 10 chars → filtré
            FakeNewsItem(title="A" * 5),        # < 10 chars → filtré
            FakeNewsItem(title="Fed raises rates sharply"),  # OK
        ]
        seed = MiroFishClient.news_to_seed(news)
        assert len(seed) == 1
        assert seed[0]["title"] == "Fed raises rates sharply"

    def test_news_to_seed_max_20_items(self):
        news = [FakeNewsItem(title=f"Breaking news item number {i}") for i in range(30)]
        seed = MiroFishClient.news_to_seed(news)
        assert len(seed) == 20

    def test_news_to_seed_truncates_title(self):
        news = [FakeNewsItem(title="A" * 600)]
        seed = MiroFishClient.news_to_seed(news)
        assert len(seed[0]["title"]) <= 500

    def test_news_to_seed_truncates_content(self):
        news = [FakeNewsItem(title="Valid title here", text="B" * 3000)]
        seed = MiroFishClient.news_to_seed(news)
        assert len(seed[0]["content"]) <= 2000

    def test_news_to_seed_uses_title_as_content_when_text_empty(self):
        news = [FakeNewsItem(title="Valid title here", text="")]
        seed = MiroFishClient.news_to_seed(news)
        assert seed[0]["content"] == "Valid title here"

    def test_news_to_seed_empty_list(self):
        assert MiroFishClient.news_to_seed([]) == []

    def test_news_to_seed_includes_source_and_date(self):
        news = [FakeNewsItem(title="Fed raises rates sharply", source="reuters", date="2025-01-01")]
        seed = MiroFishClient.news_to_seed(news)
        assert seed[0]["source"] == "reuters"
        assert "2025-01-01" in seed[0]["date"]


# ---------------------------------------------------------------------------
# _parse_response() — 3 formats
# ---------------------------------------------------------------------------


class TestParseResponse:
    def setup_method(self):
        self.client = make_client()

    def test_parse_response_format1(self):
        data = {"sentiment_index": -0.42, "panic_spread": 0.71}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index == pytest.approx(-0.42)
        assert result.panic_spread == pytest.approx(0.71)

    def test_parse_response_format2_sentiment(self):
        data = {"results": {"sentiment": 0.3}}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index == pytest.approx(0.3)

    def test_parse_response_format2_sentiment_index(self):
        data = {"results": {"sentiment_index": -0.15}}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index == pytest.approx(-0.15)

    def test_parse_response_format3_numeric_states(self):
        data = {"agent_states": [0.5, -0.3, 0.1]}
        result = self.client._parse_response(data, "market_news")
        expected = (0.5 + -0.3 + 0.1) / 3
        assert result.sentiment_index == pytest.approx(expected)

    def test_parse_response_format3_dict_states(self):
        data = {"agent_states": [{"sentiment": 0.8}, {"sentiment": -0.2}]}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index == pytest.approx(0.3)

    def test_sentiment_clamped_above_1(self):
        data = {"sentiment_index": 5.0}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index == pytest.approx(1.0)

    def test_sentiment_clamped_below_minus1(self):
        data = {"sentiment_index": -99.0}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index == pytest.approx(-1.0)

    def test_parse_unknown_format_returns_none(self):
        data = {"something_else": "value"}
        result = self.client._parse_response(data, "market_news")
        assert result.sentiment_index is None

    def test_scenario_stored_in_result(self):
        data = {"sentiment_index": 0.2}
        result = self.client._parse_response(data, "fed_rate_shock")
        assert result.scenario == "fed_rate_shock"


# ---------------------------------------------------------------------------
# simulate() — avec mock httpx
# ---------------------------------------------------------------------------


class TestSimulateWithMock:
    def _mock_health(self, client):
        """Patch health() pour retourner True."""
        return patch.object(client, "health", return_value=True)

    def test_simulate_with_mock_server_format1(self):
        client = make_client()
        seed = [{"title": "Fed hike", "content": "Rates up", "date": "", "source": "test"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"sentiment_index": 0.65, "panic_spread": 0.2}
        mock_resp.raise_for_status = MagicMock()

        with self._mock_health(client):
            with patch("httpx.Client") as mock_httpx:
                mock_httpx.return_value.__enter__.return_value.post.return_value = mock_resp
                result = client.simulate(seed, symbol="AAPL")

        assert result.sentiment_index == pytest.approx(0.65)
        assert result.panic_spread == pytest.approx(0.2)

    def test_simulate_with_mock_server_format2(self):
        client = make_client()
        seed = [{"title": "Sovereign default risk", "content": "...", "date": "", "source": "test"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"sentiment": -0.8}}
        mock_resp.raise_for_status = MagicMock()

        with self._mock_health(client):
            with patch("httpx.Client") as mock_httpx:
                mock_httpx.return_value.__enter__.return_value.post.return_value = mock_resp
                result = client.simulate(seed, symbol="SPY")

        assert result.sentiment_index == pytest.approx(-0.8)

    def test_simulate_timeout_returns_none(self):
        import httpx as httpx_mod
        client = make_client(timeout=1.0)
        seed = [{"title": "Some news item here", "content": "...", "date": "", "source": "test"}]

        with self._mock_health(client):
            with patch("httpx.Client") as mock_httpx:
                mock_httpx.return_value.__enter__.return_value.post.side_effect = (
                    httpx_mod.TimeoutException("timeout")
                )
                result = client.simulate(seed, symbol="TEST")

        assert result.sentiment_index is None

    def test_simulate_http_error_returns_none(self):
        client = make_client()
        seed = [{"title": "Some news item here", "content": "...", "date": "", "source": "test"}]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 503")

        with self._mock_health(client):
            with patch("httpx.Client") as mock_httpx:
                mock_httpx.return_value.__enter__.return_value.post.return_value = mock_resp
                result = client.simulate(seed, symbol="TEST")

        assert result.sentiment_index is None

    def test_simulate_latency_populated(self):
        client = make_client()
        seed = [{"title": "Market shock event", "content": "...", "date": "", "source": "test"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"sentiment_index": 0.1}
        mock_resp.raise_for_status = MagicMock()

        with self._mock_health(client):
            with patch("httpx.Client") as mock_httpx:
                mock_httpx.return_value.__enter__.return_value.post.return_value = mock_resp
                result = client.simulate(seed, symbol="TEST")

        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# simulate_macro_shock()
# ---------------------------------------------------------------------------


class TestSimulateMacroShock:
    def test_macro_shock_creates_synthetic_seed(self):
        client = make_client()
        # Pas de serveur → retourne None, mais ne crash pas
        result = client.simulate_macro_shock("rate_hike_200bps", magnitude=1.5, symbol="TEST")
        assert isinstance(result, SimulationResult)
        assert result.sentiment_index is None  # pas de serveur

    def test_macro_shock_with_mock(self):
        client = make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"sentiment_index": -0.7}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "health", return_value=True):
            with patch("httpx.Client") as mock_httpx:
                mock_httpx.return_value.__enter__.return_value.post.return_value = mock_resp
                result = client.simulate_macro_shock("sovereign_default", symbol="TEST")

        assert result.sentiment_index == pytest.approx(-0.7)

    def test_macro_shock_seed_contains_shock_type(self):
        """Le seed synthétique doit mentionner le shock_type."""
        captured = {}

        def fake_simulate(seed, scenario, symbol):
            captured["seed"] = seed
            return SimulationResult(sentiment_index=None, panic_spread=None)

        client = make_client()
        with patch.object(client, "simulate", side_effect=fake_simulate):
            client.simulate_macro_shock("liquidity_crisis", symbol="TEST")

        assert len(captured["seed"]) == 1
        assert "liquidity_crisis" in captured["seed"][0]["title"].lower() or \
               "LIQUIDITY CRISIS" in captured["seed"][0]["title"]


# ---------------------------------------------------------------------------
# Intégration avec SignalAggregator
# ---------------------------------------------------------------------------


class TestAggregatorMiroFish:
    def _make_prices(self):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(0)
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 300)))
        return pd.DataFrame({
            "open": close * 0.999, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, 300).astype(float),
        }, index=dates)

    def test_aggregator_skips_mirofish_by_default(self):
        """compute() sans use_mirofish=True n'appelle pas MiroFishClient."""
        from signals.aggregator import SignalAggregator
        prices = self._make_prices()
        agg = SignalAggregator(min_bars=100)

        with patch("simulation.mirofish_client.MiroFishClient") as mock_cls:
            vector = agg.compute(prices, symbol="AAPL")

        mock_cls.assert_not_called()
        assert vector.mirofish_sentiment is None

    def test_aggregator_uses_manual_sentiment(self):
        """compute(mirofish_sentiment=-0.5) injecte directement sans simulation."""
        from signals.aggregator import SignalAggregator
        prices = self._make_prices()
        agg = SignalAggregator(min_bars=100)

        vector = agg.compute(prices, symbol="AAPL", mirofish_sentiment=-0.5)
        assert vector.mirofish_sentiment == pytest.approx(-0.5)

    def test_aggregator_mirofish_skipped_when_server_down(self):
        """use_mirofish=True mais serveur down → sentiment reste None."""
        from signals.aggregator import SignalAggregator
        prices = self._make_prices()
        agg = SignalAggregator(min_bars=100)
        news = [FakeNewsItem(title="Fed raises rates sharply")]

        # MiroFishClient.health() retourne False → pas de simulation
        with patch("simulation.mirofish_client.MiroFishClient.health", return_value=False):
            vector = agg.compute(
                prices, symbol="AAPL",
                use_mirofish=True, mirofish_news=news,
            )

        assert vector.mirofish_sentiment is None

    def test_aggregator_fills_sentiment_when_mirofish_active(self):
        """use_mirofish=True + serveur mock → vector.mirofish_sentiment rempli."""
        from signals.aggregator import SignalAggregator
        from simulation.mirofish_client import SimulationResult
        prices = self._make_prices()
        agg = SignalAggregator(min_bars=100)
        news = [FakeNewsItem(title="Fed raises rates sharply")]

        fake_result = SimulationResult(sentiment_index=0.42, panic_spread=0.1)
        with patch("simulation.mirofish_client.MiroFishClient.health", return_value=True):
            with patch("simulation.mirofish_client.MiroFishClient.simulate", return_value=fake_result):
                vector = agg.compute(
                    prices, symbol="AAPL",
                    use_mirofish=True, mirofish_news=news,
                )

        assert vector.mirofish_sentiment == pytest.approx(0.42)

    def test_aggregator_no_news_logs_skip(self):
        """use_mirofish=True sans news → sentiment=None, pas de crash."""
        from signals.aggregator import SignalAggregator
        prices = self._make_prices()
        agg = SignalAggregator(min_bars=100)

        with patch("simulation.mirofish_client.MiroFishClient.health", return_value=True):
            vector = agg.compute(
                prices, symbol="AAPL",
                use_mirofish=True,
                mirofish_news=[],
            )

        assert vector.mirofish_sentiment is None
