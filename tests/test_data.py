"""
tests/test_data.py
------------------
Tests du client OpenBB.

Pour lancer :
    uv run pytest tests/test_data.py -v

Les tests marqués @pytest.mark.integration nécessitent
le serveur OpenBB local actif (openbb-api).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.client import NewsItem, OpenBBClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Client avec HTTP mocké — pas besoin du serveur OpenBB."""
    client = OpenBBClient.__new__(OpenBBClient)
    client.base_url = "http://mock"
    client.timeout = 5.0
    client._client = MagicMock()
    return client


MOCK_OHLCV_RESPONSE = {
    "results": [
        {
            "date": "2024-01-02",
            "open": 185.0,
            "high": 187.0,
            "low": 184.0,
            "close": 186.0,
            "volume": 50_000_000,
        },
        {
            "date": "2024-01-03",
            "open": 186.0,
            "high": 188.0,
            "low": 185.0,
            "close": 187.5,
            "volume": 45_000_000,
        },
        {
            "date": "2024-01-04",
            "open": 187.0,
            "high": 189.0,
            "low": 186.0,
            "close": 185.0,
            "volume": 60_000_000,
        },
    ]
}


# ---------------------------------------------------------------------------
# Tests unitaires (sans serveur)
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_standard_columns(self):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1_000_000, 1_100_000],
            }
        )
        result = OpenBBClient._normalize_ohlcv(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert set(["open", "high", "low", "close", "volume"]).issubset(result.columns)
        assert len(result) == 2

    def test_normalize_drops_nan_close(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "open": [100.0, None],
                "high": [102.0, None],
                "low": [99.0, None],
                "close": [None, None],
                "volume": [1_000_000, None],
            }
        )
        result = OpenBBClient._normalize_ohlcv(df)
        assert len(result) == 0

    def test_normalize_sorted_index(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
                "open": [103.0, 101.0, 102.0],
                "high": [104.0, 102.0, 103.0],
                "low": [102.0, 100.0, 101.0],
                "close": [103.5, 101.5, 102.5],
                "volume": [1e6, 1e6, 1e6],
            }
        )
        result = OpenBBClient._normalize_ohlcv(df)
        assert result.index.is_monotonic_increasing


class TestOHLCV:
    def test_ohlcv_returns_dataframe(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_OHLCV_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client._client.get.return_value = mock_resp

        df = mock_client.ohlcv("AAPL", start="2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "close" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_ohlcv_default_start_is_one_year(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_client._client.get.return_value = mock_resp

        mock_client.ohlcv("AAPL")

        call_params = mock_client._client.get.call_args
        params = call_params[1].get("params", {})
        assert "start_date" in params

    def test_ohlcv_raises_on_connection_error(self, mock_client):
        import httpx

        mock_client._client.get.side_effect = httpx.ConnectError("refused")
        with pytest.raises(ConnectionError, match="openbb-api"):
            mock_client.ohlcv("AAPL")


class TestNews:
    def test_news_returns_list_of_newsitem(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "date": "2024-01-01T12:00:00",
                    "title": "Apple hits ATH",
                    "text": "Apple stock...",
                    "url": "https://example.com",
                    "source": "Reuters",
                },
            ]
        }
        mock_client._client.get.return_value = mock_resp

        items = mock_client.news("AAPL", limit=1)
        assert len(items) == 1
        assert isinstance(items[0], NewsItem)
        assert items[0].title == "Apple hits ATH"

    def test_news_as_df_has_correct_columns(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"date": "2024-01-01T12:00:00", "title": "Test", "source": "BB"},
            ]
        }
        mock_client._client.get.return_value = mock_resp

        df = mock_client.news_as_df("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert "title" in df.columns
        assert "date" in df.columns


class TestHealth:
    def test_health_true_on_200(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client._client.get.return_value = mock_resp
        assert mock_client.health() is True

    def test_health_false_on_exception(self, mock_client):
        mock_client._client.get.side_effect = Exception("unreachable")
        assert mock_client.health() is False


# ---------------------------------------------------------------------------
# Tests d'intégration (nécessitent openbb-api actif)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """Lance avec : uv run pytest -m integration"""

    def test_server_is_up(self):
        client = OpenBBClient()
        assert client.health(), "Lance 'openbb-api' avant les tests d'intégration"

    def test_real_ohlcv_aapl(self):
        client = OpenBBClient()
        df = client.ohlcv("AAPL", start="2024-01-01", end="2024-06-01")
        assert len(df) > 50
        assert df["close"].notna().all()
        assert df.index.is_monotonic_increasing

    def test_real_macro_dashboard(self):
        client = OpenBBClient()
        df = client.macro_dashboard()
        assert "vix" in df.columns
        assert "fed_funds" in df.columns

    def test_real_news_aapl(self):
        client = OpenBBClient()
        items = client.news("AAPL", limit=5)
        assert len(items) > 0
        assert all(isinstance(i, NewsItem) for i in items)
