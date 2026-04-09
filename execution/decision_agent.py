"""
execution/decision_agent.py
---------------------------
Decision Agent LLM basé sur Claude (claude-sonnet-4-6).

Prend un SignalVector, appelle l'API Claude, et retourne un ordre structuré.

Règle hard : confidence < 0.6 → direction = "FLAT" automatiquement.

Usage :
    agent = DecisionAgent()
    order = agent.decide(signal_vector)
    print(order)
    # → TradeOrder(direction="LONG", confidence=0.78, ...)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from execution.risk import Portfolio, RiskEngine, RiskLimits
from signals.aggregator import SignalVector

load_dotenv()

Direction = Literal["LONG", "SHORT", "FLAT"]

SYSTEM_PROMPT = """You are a quantitative trading decision agent. You receive a signal vector
computed from momentum, value, quality, and volatility factors, and you must output a trading
decision as valid JSON.

Rules:
- Analyze all available signals holistically.
- Output ONLY a JSON object with these exact fields:
  {
    "direction": "LONG" | "SHORT" | "FLAT",
    "confidence": <float 0.0-1.0>,
    "entry": <float, suggested entry price as % offset from current, e.g. 0.0 for market>,
    "stop_loss": <float, stop distance as % of price, e.g. 0.02 for 2%>,
    "take_profit": <float, target as % of price, e.g. 0.04 for 4%>,
    "size_pct": <float 0.0-1.0, fraction of portfolio to allocate>,
    "rationale": <string, concise explanation in 1-2 sentences>
  }
- If confidence < 0.6, you MUST set direction to "FLAT" and size_pct to 0.0.
- Be conservative: prefer FLAT when signals conflict or data quality is low.
- size_pct should scale with confidence: high confidence → larger size (max 0.05 = 5%).
- Output ONLY the JSON object, no markdown, no explanation outside the JSON."""


@dataclass
class TradeOrder:
    """Ordre de trading structuré retourné par le Decision Agent."""

    symbol: str
    direction: Direction
    confidence: float
    entry: float = 0.0           # % offset from current price (0.0 = market order)
    stop_loss: float = 0.02      # stop distance as % of price
    take_profit: float = 0.04    # take-profit distance as % of price
    size_pct: float = 0.0        # fraction of portfolio (0→1)
    rationale: str = ""

    def __post_init__(self) -> None:
        # Hard rule: confidence < 0.6 → FLAT
        if self.confidence < 0.6:
            self.direction = "FLAT"
            self.size_pct = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def is_active(self) -> bool:
        return self.direction != "FLAT"


class DecisionAgent:
    """
    Agent de décision utilisant Claude pour interpréter un SignalVector.

    Exemple :
        from signals.aggregator import SignalAggregator
        from execution.decision_agent import DecisionAgent

        agent = DecisionAgent()
        vector = agg.compute(prices, symbol="AAPL")
        order = agent.decide(vector)
        print(order)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic()  # lit ANTHROPIC_API_KEY depuis l'env automatiquement

    def decide(self, vector: SignalVector) -> TradeOrder:
        """
        Analyse le vecteur de signal et retourne un ordre de trading.

        Args:
            vector: SignalVector calculé par SignalAggregator

        Returns:
            TradeOrder avec direction, confidence, sizing, et rationale
        """
        prompt = vector.to_prompt()
        logger.info(f"DecisionAgent.decide({vector.symbol}) — calling {self.model}")

        raw_json = self._call_claude(prompt)
        order = self._parse_order(raw_json, vector.symbol)

        logger.info(
            f"{vector.symbol} → {order.direction} "
            f"conf={order.confidence:.2f} size={order.size_pct:.2%} | {order.rationale[:60]}"
        )
        return order

    def _call_claude(self, signal_prompt: str) -> str:
        """Appelle l'API Claude et retourne la réponse brute."""
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": signal_prompt,
                }
            ],
        )
        return message.content[0].text

    def decide_with_risk(
        self,
        vector: SignalVector,
        portfolio: Portfolio,
        recent_returns: list[float] | None = None,
        limits: RiskLimits | None = None,
    ) -> TradeOrder | None:
        """
        Pipeline complet : décision LLM → validation risque.
        Retourne None si le risk engine bloque l'ordre.
        """
        order = self.decide(vector)
        engine = RiskEngine(portfolio, limits or RiskLimits())
        return engine.validate(order, recent_returns)

    def _parse_order(self, raw: str, symbol: str) -> TradeOrder:
        """Parse la réponse JSON de Claude en TradeOrder."""
        try:
            # Strip markdown code blocks if present
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(text)

            direction = data.get("direction", "FLAT")
            if direction not in ("LONG", "SHORT", "FLAT"):
                logger.warning(f"Direction invalide '{direction}' → FLAT")
                direction = "FLAT"

            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            size_pct = float(data.get("size_pct", 0.0))
            size_pct = max(0.0, min(0.05, size_pct))  # cap at 5%

            return TradeOrder(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry=float(data.get("entry", 0.0)),
                stop_loss=float(data.get("stop_loss", 0.02)),
                take_profit=float(data.get("take_profit", 0.04)),
                size_pct=size_pct,
                rationale=str(data.get("rationale", "")),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Échec parsing réponse Claude: {e}\nRaw: {raw[:200]}")
            return TradeOrder(
                symbol=symbol,
                direction="FLAT",
                confidence=0.0,
                rationale=f"Parse error: {e}",
            )
