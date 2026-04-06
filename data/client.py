"""
data/client.py
--------------
Wrapper unifié autour de l'API REST OpenBB.
Tous les modules du projet passent par cette classe — jamais directement
par yfinance, Alpha Vantage, ou autre source tierce.

Usage :
    client = OpenBBClient()
    df = client.ohlcv("AAPL", start="2022-01-01")
    news = client.news("AAPL", limit=50)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Literal

import httpx
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

OPENBB_API_URL = os.getenv("OPENBB_API_URL", "http://127.0.0.1:6900")


# ---------------------------------------------------------------------------
# Modèles de réponse (Pydantic)
# ---------------------------------------------------------------------------


class OHLCVBar(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewsItem(BaseModel):
    date: datetime
    title: str
    text: str = ""
    url: str = ""
    source: str = ""
    sentiment: float | None = None  # -1.0 → +1.0


class MacroSeries(BaseModel):
    series_id: str
    title: str
    data: dict[str, float]  # date → valeur


class FundamentalSnapshot(BaseModel):
    symbol: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    market_cap: float | None = None
    sector: str | None = None


# ---------------------------------------------------------------------------
# Client principal
# ---------------------------------------------------------------------------


@dataclass
class OpenBBClient:
    """
    Client HTTP autour du serveur OpenBB local (FastAPI sur port 6900).

    Toutes les méthodes retournent des DataFrames pandas prêts à l'emploi
    ou des modèles Pydantic selon le contexte.
    """

    base_url: str = field(default_factory=lambda: OPENBB_API_URL)
    timeout: float = 30.0
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        logger.info(f"OpenBBClient → {self.base_url}")

    def _get(self, path: str, **params) -> dict:
        """GET générique avec logging et gestion d'erreur."""
        try:
            r = self._client.get(path, params={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"OpenBB API non joignable sur {self.base_url}. "
                "Lance 'd'abord : openbb-api'"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} sur {path}: {e.response.text}")
            raise

    # ------------------------------------------------------------------
    # Prix OHLCV
    # ------------------------------------------------------------------

    def ohlcv(
        self,
        symbol: str,
        start: str | date | None = None,
        end: str | date | None = None,
        interval: Literal["1d", "1wk", "1mo"] = "1d",
        provider: str = "yfinance",
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame OHLCV avec index DatetimeIndex.

        Exemple :
            df = client.ohlcv("AAPL", start="2020-01-01")
            df.tail()
        """
        if start is None:
            start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

        raw = self._get(
            "/api/v1/equity/price/historical",
            symbol=symbol,
            start_date=str(start),
            end_date=str(end) if end else None,
            interval=interval,
            provider=provider,
        )

        df = pd.DataFrame(raw.get("results", raw))
        df = self._normalize_ohlcv(df)
        logger.debug(f"ohlcv({symbol}) → {len(df)} barres")
        return df

    def ohlcv_multi(
        self,
        symbols: list[str],
        start: str | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """Retourne un dict symbol → DataFrame."""
        return {s: self.ohlcv(s, start=start, **kwargs) for s in symbols}

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise les noms de colonnes et l'index."""
        col_map = {
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "close" not in df.columns:
            return df
        return df.dropna(subset=["close"])

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def news(
        self,
        symbol: str,
        limit: int = 50,
        provider: str = "benzinga",
    ) -> list[NewsItem]:
        """
        Retourne les dernières news d'un symbole.
        Format utile comme seed pour MiroFish.

        Exemple :
            items = client.news("AAPL", limit=20)
            seed = [{"title": n.title, "text": n.text} for n in items]
        """
        raw = self._get(
            "/api/v1/news/company",
            symbols=symbol,
            limit=limit,
            provider=provider,
        )
        items = []
        for r in raw.get("results", []):
            try:
                items.append(NewsItem(
                    date=r.get("date", datetime.now()),
                    title=r.get("title", ""),
                    text=r.get("text", r.get("body", "")),
                    url=r.get("url", ""),
                    source=r.get("source", ""),
                ))
            except Exception as e:
                logger.warning(f"News parsing error: {e}")
        logger.debug(f"news({symbol}) → {len(items)} articles")
        return items

    def news_as_df(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Version DataFrame de news()."""
        items = self.news(symbol, **kwargs)
        return pd.DataFrame([i.model_dump() for i in items])

    # ------------------------------------------------------------------
    # Données macro (FRED)
    # ------------------------------------------------------------------

    def fred(self, series_id: str) -> pd.Series:
        """
        Récupère une série FRED.

        Exemples de series_id :
            CPIAUCSL  → inflation CPI
            FEDFUNDS  → taux fed funds
            T10Y2Y    → spread 10Y-2Y (courbe des taux)
            VIXCLS    → VIX
            DGS10     → taux 10 ans

        Retourne une pd.Series avec index DatetimeIndex.
        """
        raw = self._get(
            "/api/v1/economy/fred_series",
            symbol=series_id,
            provider="fred",
        )
        data = raw.get("results", raw)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            val_col = next((c for c in df.columns if "value" in c.lower()), None)
            if date_col and val_col:
                s = df.set_index(date_col)[val_col]
                s.index = pd.to_datetime(s.index)
                s.name = series_id
                return s.sort_index().apply(pd.to_numeric, errors="coerce")
        raise ValueError(f"Format inattendu pour FRED {series_id}")

    def macro_dashboard(self) -> pd.DataFrame:
        """
        Snapshot macro en une ligne : CPI, taux Fed, 10Y-2Y, VIX.
        Utile comme feature macro dans le vecteur signal.
        """
        series = {
            "cpi_yoy": "CPIAUCSL",
            "fed_funds": "FEDFUNDS",
            "spread_10y2y": "T10Y2Y",
            "vix": "VIXCLS",
        }
        latest = {}
        for name, sid in series.items():
            try:
                s = self.fred(sid)
                latest[name] = float(s.dropna().iloc[-1])
            except Exception as e:
                logger.warning(f"Macro {sid} indisponible: {e}")
                latest[name] = None

        return pd.DataFrame([latest], index=[pd.Timestamp.today().date()])

    # ------------------------------------------------------------------
    # Fondamentaux
    # ------------------------------------------------------------------

    def fundamentals(self, symbol: str) -> FundamentalSnapshot:
        """Ratios fondamentaux : P/E, P/B, ROE, D/E, market cap."""
        try:
            raw = self._get(
                "/api/v1/equity/fundamental/ratios",
                symbol=symbol,
                period="annual",
                limit=1,
                provider="yfinance",
            )
            r = raw.get("results", [{}])[0]
            return FundamentalSnapshot(
                symbol=symbol,
                pe_ratio=r.get("pe_ratio"),
                pb_ratio=r.get("price_to_book"),
                roe=r.get("return_on_equity"),
                debt_to_equity=r.get("debt_equity_ratio"),
                market_cap=r.get("market_cap"),
            )
        except Exception as e:
            logger.warning(f"Fondamentaux {symbol} indisponibles: {e}")
            return FundamentalSnapshot(symbol=symbol)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """Vérifie que le serveur OpenBB répond."""
        try:
            r = self._client.get("/")
            return r.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Singleton global (optionnel)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> OpenBBClient:
    """Retourne une instance partagée du client."""
    return OpenBBClient()
