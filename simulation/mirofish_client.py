"""
simulation/mirofish_client.py
------------------------------
Client HTTP vers MiroFish (localhost:5001).

Flux :
    news = client.news("AAPL", limit=20)
    seed = MiroFishClient.news_to_seed(news)
    result = MiroFishClient().simulate(seed, scenario="market_shock")
    vector.mirofish_sentiment = result.sentiment_index

MiroFish est optionnel — si le serveur n'est pas actif,
sentiment_index=None et le pipeline continue sans crash.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

MIROFISH_URL = os.getenv("MIROFISH_API_URL", "http://127.0.0.1:5001")


@dataclass
class SimulationResult:
    sentiment_index: float | None    # -1.0 (très bearish) → +1.0 (très bullish)
    panic_spread: float | None       # 0.0 → 1.0 (propagation de la panique)
    n_agents: int = 0
    n_rounds: int = 0
    scenario: str = ""
    raw: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class MiroFishClient:
    """
    Client HTTP vers MiroFish.
    Non bloquant : si serveur absent → SimulationResult(sentiment_index=None)
    """

    def __init__(
        self,
        base_url: str = MIROFISH_URL,
        timeout: float = 60.0,
        n_agents: int = 200,
        n_rounds: int = 10,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def health(self) -> bool:
        """Vérifie que MiroFish répond."""
        try:
            r = httpx.get(f"{self.base_url}/", timeout=3.0)
            return r.status_code < 500
        except Exception:
            return False

    @staticmethod
    def news_to_seed(news_items: list) -> list[dict]:
        """
        Convertit les NewsItem d'OpenBB en seed MiroFish.
        Filtre les articles vides ou trop courts.
        """
        seed = []
        for item in news_items:
            title = getattr(item, "title", str(item))
            content = getattr(item, "text", getattr(item, "content", ""))
            if len(title) < 10:
                continue
            seed.append({
                "title": title[:500],
                "content": content[:2000] if content else title,
                "date": str(getattr(item, "date", datetime.now().isoformat())),
                "source": getattr(item, "source", "unknown"),
            })
        return seed[:20]

    def simulate(
        self,
        seed: list[dict],
        scenario: str = "market_news",
        symbol: str = "UNKNOWN",
    ) -> SimulationResult:
        """
        Lance une simulation MiroFish et retourne le sentiment agrégé.

        Args:
            seed     : liste de news articles (output de news_to_seed())
            scenario : "market_news" | "fed_rate_shock" | "credit_event" | "geopolitical"
            symbol   : ticker pour les logs

        Returns:
            SimulationResult avec sentiment_index ∈ [-1.0, +1.0]
            sentiment_index=None si serveur indisponible ou erreur
        """
        t0 = time.perf_counter()

        if not seed:
            logger.warning(f"{symbol}: MiroFish seed vide — skip simulation")
            return SimulationResult(sentiment_index=None, panic_spread=None)

        if not self.health():
            logger.info(f"{symbol}: MiroFish non disponible sur {self.base_url}")
            return SimulationResult(sentiment_index=None, panic_spread=None)

        try:
            payload = {
                "seed": seed,
                "scenario": scenario,
                "n_agents": self.n_agents,
                "rounds": self.n_rounds,
            }

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/simulate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            result = self._parse_response(data, scenario)
            result.latency_ms = (time.perf_counter() - t0) * 1000

            if result.sentiment_index is not None:
                logger.info(
                    f"{symbol} → mirofish sentiment={result.sentiment_index:+.3f} "
                    f"panic={result.panic_spread:.2f} "
                    f"({self.n_agents} agents, {result.latency_ms:.0f}ms)"
                )
            else:
                logger.warning(f"{symbol}: MiroFish simulation failed")
            return result

        except httpx.TimeoutException:
            logger.warning(f"{symbol}: MiroFish timeout après {self.timeout}s")
            return SimulationResult(sentiment_index=None, panic_spread=None)
        except Exception as e:
            logger.warning(f"{symbol}: MiroFish error — {e}")
            return SimulationResult(sentiment_index=None, panic_spread=None)

    def _parse_response(self, data: dict, scenario: str) -> SimulationResult:
        """
        Parse la réponse JSON de MiroFish.
        Essaie plusieurs formats possibles de l'API.
        """
        sentiment = None

        # Format 1 : {"sentiment_index": -0.42, "panic_spread": 0.71}
        if "sentiment_index" in data:
            sentiment = float(data["sentiment_index"])

        # Format 2 : {"results": {"sentiment": -0.42}}
        elif "results" in data and isinstance(data["results"], dict):
            r = data["results"]
            sentiment = float(r.get("sentiment", r.get("sentiment_index", 0.0)))

        # Format 3 : {"agent_states": [...]} → moyenne des états
        elif "agent_states" in data:
            states = data["agent_states"]
            if states:
                numeric = []
                for s in states:
                    if isinstance(s, (int, float)):
                        numeric.append(float(s))
                    elif isinstance(s, dict) and "sentiment" in s:
                        numeric.append(float(s["sentiment"]))
                if numeric:
                    sentiment = sum(numeric) / len(numeric)

        # Clamp [-1, +1]
        if sentiment is not None:
            sentiment = max(-1.0, min(1.0, sentiment))

        return SimulationResult(
            sentiment_index=sentiment,
            panic_spread=float(data.get("panic_spread", 0.0)),
            n_agents=int(data.get("n_agents", self.n_agents)),
            n_rounds=int(data.get("rounds", self.n_rounds)),
            scenario=scenario,
            raw=data,
        )

    def simulate_macro_shock(
        self,
        shock_type: str,
        magnitude: float = 1.0,
        symbol: str = "UNKNOWN",
    ) -> SimulationResult:
        """
        Simule un choc macro sans seed textuel.

        Args:
            shock_type : "rate_hike_200bps" | "sovereign_default" | "liquidity_crisis"
            magnitude  : intensité du choc (1.0 = standard)
        """
        synthetic_seed = [{
            "title": f"BREAKING: {shock_type.replace('_', ' ').upper()}",
            "content": (
                f"Major {shock_type} event with magnitude {magnitude}x. "
                f"Market participants are reassessing risk exposure."
            ),
            "date": datetime.now().isoformat(),
            "source": "synthetic_stress_test",
        }]
        return self.simulate(synthetic_seed, scenario=shock_type, symbol=symbol)
