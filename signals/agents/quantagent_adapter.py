"""
signals/agents/quantagent_adapter.py
-------------------------------------
Adaptateur entre QuantAgent (submodule) et le pipeline QuantAI.

Prend un DataFrame OHLCV OpenBB, appelle TradingGraph de QuantAgent,
et retourne un AgentSignal avec agent_bias (-1.0 → +1.0).

Règles :
- Si le submodule est absent ou la clé API manque → agent_bias=None, ne bloque pas
- Timeout 30s strict via ThreadPoolExecutor
- Ne lève jamais d'exception vers l'appelant

Usage :
    adapter = QuantAgentAdapter(llm_provider="anthropic")
    if adapter.is_available():
        signal = adapter.analyze(prices_df, symbol="AAPL")
        print(signal.agent_bias)  # float -1.0 → +1.0
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

# Chemin vers le submodule QuantAgent
_QUANTAGENT_PATH = Path(__file__).parent / "quantagent"

# Provider → model par défaut
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "qwen": "qwen-vl-max-latest",
    "minimax": "MiniMax-M2.7",
}

_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Résultat de l'analyse QuantAgent
# ---------------------------------------------------------------------------


@dataclass
class AgentSignal:
    """
    Signal retourné par QuantAgentAdapter.analyze().

    agent_bias : float entre -1.0 et +1.0, ou None si l'analyse a échoué.
    """

    agent_bias: float | None  # -1.0 (fort SHORT) → +1.0 (fort LONG)
    direction: str  # "LONG" | "SHORT" | "FLAT" | "UNKNOWN"
    indicator_report: str | None = None
    pattern_report: str | None = None
    trend_report: str | None = None
    confidence: float | None = None  # 0.0 → 1.0 (déduit du risk_reward_ratio)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Adaptateur principal
# ---------------------------------------------------------------------------


class QuantAgentAdapter:
    """
    Pont entre QuantAgent (multi-agent LangGraph) et SignalAggregator.

    Args:
        llm_provider : "anthropic" (défaut), "openai", "qwen", "minimax"
        model        : override du modèle (None = défaut par provider)
        api_key      : override de la clé API (None = depuis .env / env)
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.llm_provider = llm_provider
        self.model = model or _DEFAULT_MODELS.get(llm_provider, "gpt-4o")
        self.api_key = api_key

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Vérifie que le submodule est présent et la clé API configurée.
        Ne tente aucun appel réseau.
        """
        if not _QUANTAGENT_PATH.exists():
            logger.warning("QuantAgent submodule absent — lancez : git submodule update --init --recursive")
            return False

        if not ((_QUANTAGENT_PATH / "trading_graph.py").exists()):
            logger.warning("QuantAgent : trading_graph.py introuvable dans le submodule")
            return False

        if not self._has_api_key():
            logger.warning(f"QuantAgent : clé API manquante pour le provider '{self.llm_provider}'")
            return False

        return True

    def analyze(
        self,
        prices: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> AgentSignal:
        """
        Lance l'analyse QuantAgent sur un DataFrame OHLCV.

        Args:
            prices : DataFrame avec colonnes open/high/low/close/volume
                     et index DatetimeIndex (format OpenBB)
            symbol : ticker (ex. "AAPL")

        Returns:
            AgentSignal (agent_bias=None si l'analyse a échoué)
        """
        t0 = time.monotonic()

        if not self.is_available():
            return AgentSignal(agent_bias=None, direction="UNKNOWN", latency_ms=0.0)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_analysis, prices, symbol)
                result = future.result(timeout=_TIMEOUT_SECONDS)
            result.latency_ms = (time.monotonic() - t0) * 1000
            return result

        except FuturesTimeoutError:
            logger.warning(f"QuantAgent timeout ({_TIMEOUT_SECONDS}s) pour {symbol}")
            return AgentSignal(
                agent_bias=None,
                direction="UNKNOWN",
                latency_ms=_TIMEOUT_SECONDS * 1000,
            )
        except Exception as e:
            logger.warning(f"QuantAgent erreur pour {symbol}: {e}")
            return AgentSignal(
                agent_bias=None,
                direction="UNKNOWN",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    # ------------------------------------------------------------------
    # Logique interne
    # ------------------------------------------------------------------

    def _has_api_key(self) -> bool:
        if self.api_key:
            return True
        env_keys = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "minimax": "MINIMAX_API_KEY",
        }
        env_var = env_keys.get(self.llm_provider)
        return bool(env_var and os.environ.get(env_var))

    def _run_analysis(self, prices: pd.DataFrame, symbol: str) -> AgentSignal:
        """Exécuté dans un thread séparé (pour le timeout)."""
        # Injection du path submodule
        if str(_QUANTAGENT_PATH) not in sys.path:
            sys.path.insert(0, str(_QUANTAGENT_PATH))

        import static_util  # type: ignore
        from trading_graph import TradingGraph  # type: ignore

        # Prépare les données (30 dernières bougies)
        kline_data = self._to_kline_dict(prices)

        # Config QuantAgent
        config = self._build_config()

        trading_graph = TradingGraph(config=config)

        # Génère les images pour les agents Pattern et Trend
        try:
            p_image = static_util.generate_kline_image(kline_data)
            t_image = static_util.generate_trend_image(kline_data)
            pattern_image = p_image.get("pattern_image", "")
            trend_image = t_image.get("trend_image", "")
        except Exception as e:
            logger.warning(f"QuantAgent : génération image échouée ({e}) — on continue sans image")
            pattern_image = ""
            trend_image = ""

        initial_state = {
            "kline_data": kline_data,
            "analysis_results": None,
            "messages": [],
            "time_frame": "1day",
            "stock_name": symbol,
            "pattern_image": pattern_image,
            "trend_image": trend_image,
        }

        final_state = trading_graph.graph.invoke(initial_state)
        return self._parse_state(final_state)

    def _to_kline_dict(self, prices: pd.DataFrame) -> dict:
        """
        Convertit un DataFrame OHLCV OpenBB au format kline_data de QuantAgent.

        Entrée : colonnes open/high/low/close/volume, index DatetimeIndex
        Sortie : dict avec Datetime/Open/High/Low/Close (30 dernières lignes)
        """
        df = prices.copy().tail(30)

        # Normalise les noms de colonnes (OpenBB retourne en minuscules)
        col_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        df = df.reset_index()

        # Colonne de dates : index peut s'appeler "date" ou "datetime"
        date_col = None
        for candidate in ["date", "datetime", "Date", "Datetime", df.columns[0]]:
            if candidate in df.columns:
                date_col = candidate
                break

        kline: dict = {}
        if date_col:
            kline["Datetime"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                kline[col] = df[col].tolist()

        return kline

    def _build_config(self) -> dict:
        config: dict = {
            "agent_llm_provider": self.llm_provider,
            "agent_llm_model": self.model,
            "agent_llm_temperature": 0.1,
            "graph_llm_provider": self.llm_provider,
            "graph_llm_model": self.model,
            "graph_llm_temperature": 0.1,
        }
        if self.api_key:
            key_field = {
                "anthropic": "anthropic_api_key",
                "openai": "api_key",
                "qwen": "qwen_api_key",
                "minimax": "minimax_api_key",
            }.get(self.llm_provider, "api_key")
            config[key_field] = self.api_key
        return config

    def _parse_state(self, state: dict) -> AgentSignal:
        """Parse le final_state retourné par TradingGraph.graph.invoke()."""
        raw_decision = state.get("final_trade_decision", "")
        indicator_report = state.get("indicator_report")
        pattern_report = state.get("pattern_report")
        trend_report = state.get("trend_report")

        direction, bias, confidence = self._parse_decision(raw_decision)

        return AgentSignal(
            agent_bias=bias,
            direction=direction,
            indicator_report=indicator_report,
            pattern_report=pattern_report,
            trend_report=trend_report,
            confidence=confidence,
        )

    @staticmethod
    def _parse_decision(raw: str) -> tuple[str, float | None, float | None]:
        """
        Parse final_trade_decision (string JSON ou texte brut).

        Returns:
            (direction, agent_bias, confidence)
        """
        if not raw:
            return "UNKNOWN", None, None

        # Tentative de parse JSON
        parsed: dict | None = None
        try:
            # QuantAgent peut entourer le JSON de backticks
            clean = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            pass

        if parsed and isinstance(parsed, dict):
            decision_raw = str(parsed.get("decision", "")).upper()
            rr_ratio = parsed.get("risk_reward_ratio")
        else:
            # Fallback : cherche "LONG" ou "SHORT" dans le texte brut
            upper = raw.upper()
            if "LONG" in upper:
                decision_raw = "LONG"
            elif "SHORT" in upper:
                decision_raw = "SHORT"
            else:
                decision_raw = "UNKNOWN"
            rr_ratio = None

        # Convertit risk_reward_ratio en confidence (plage 1.2–1.8 → 0.0–1.0)
        confidence: float | None = None
        if rr_ratio is not None:
            try:
                rr = float(rr_ratio)
                confidence = round(min(1.0, max(0.0, (rr - 1.2) / 0.6)), 3)
            except (TypeError, ValueError):
                pass

        # Mappe direction → agent_bias
        if decision_raw == "LONG":
            direction = "LONG"
            # Echelle : 0.3 (rr=1.2) → 1.0 (rr=1.8+)
            if confidence is not None:
                bias = round(0.3 + confidence * 0.7, 3)
            else:
                bias = 0.7
        elif decision_raw == "SHORT":
            direction = "SHORT"
            if confidence is not None:
                bias = round(-(0.3 + confidence * 0.7), 3)
            else:
                bias = -0.7
        elif decision_raw in ("FLAT", "HOLD", "NEUTRAL"):
            direction = "FLAT"
            bias = 0.0
        else:
            direction = "UNKNOWN"
            bias = None

        return direction, bias, confidence
