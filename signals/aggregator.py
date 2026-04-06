"""
signals/aggregator.py
---------------------
Agrège tous les facteurs en un vecteur de signal unique.

C'est le contrat d'interface entre la couche "research"
et la couche "décision". Le vecteur signal a toujours
les mêmes clés — les modules en aval ne savent pas
d'où viennent les valeurs.

Usage :
    agg = SignalAggregator()
    vector = agg.compute(prices, symbol="AAPL")
    print(vector)
    # → SignalVector(momentum=0.72, value=-0.31, quality=0.58, ...)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from .factors.momentum import (
    MomentumFactor,
    RiskAdjustedMomentum,
    TrendStrength,
    composite_momentum,
)
from .factors.value_quality_vol import (
    QualityFactor,
    ValueFactor,
    VolatilityFactor,
)


# ---------------------------------------------------------------------------
# Vecteur de signal — contrat d'interface
# ---------------------------------------------------------------------------


@dataclass
class SignalVector:
    """
    Vecteur de signal normalisé.
    Toutes les valeurs sont des z-scores (mean=0, std=1)
    sauf confidence (0→1) et direction.
    """

    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Facteurs quantitatifs
    momentum_3m: float | None = None
    momentum_12m: float | None = None
    momentum_composite: float | None = None
    risk_adj_momentum: float | None = None
    trend_strength: float | None = None

    value: float | None = None
    quality: float | None = None
    low_volatility: float | None = None

    # Enrichissement externe (rempli plus tard)
    mirofish_sentiment: float | None = None   # MiroFish
    agent_bias: float | None = None           # QuantAgent

    # Macro (OpenBB FRED)
    vix: float | None = None
    spread_10y2y: float | None = None

    # Méta
    data_quality: float = 1.0   # 0→1, baisse si données manquantes
    n_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        """Sérialise pour le Decision Agent LLM."""
        lines = [f"Signal vector — {self.symbol} @ {self.timestamp.isoformat()}"]
        lines.append("")
        lines.append("MOMENTUM")
        lines.append(f"  momentum_3m          : {_fmt(self.momentum_3m)}")
        lines.append(f"  momentum_12m         : {_fmt(self.momentum_12m)}")
        lines.append(f"  momentum_composite   : {_fmt(self.momentum_composite)}")
        lines.append(f"  risk_adj_momentum    : {_fmt(self.risk_adj_momentum)}")
        lines.append(f"  trend_strength       : {_fmt(self.trend_strength)}")
        lines.append("")
        lines.append("VALUE / QUALITY / RISK")
        lines.append(f"  value                : {_fmt(self.value)}")
        lines.append(f"  quality              : {_fmt(self.quality)}")
        lines.append(f"  low_volatility       : {_fmt(self.low_volatility)}")
        lines.append("")
        lines.append("EXTERNE")
        lines.append(f"  mirofish_sentiment   : {_fmt(self.mirofish_sentiment)}")
        lines.append(f"  agent_bias           : {_fmt(self.agent_bias)}")
        lines.append(f"  vix                  : {_fmt(self.vix)}")
        lines.append(f"  spread_10y2y         : {_fmt(self.spread_10y2y)}")
        lines.append("")
        lines.append(f"data_quality : {self.data_quality:.2f}  |  n_bars : {self.n_bars}")
        return "\n".join(lines)

    @property
    def composite_score(self) -> float | None:
        """Score agrégé simple — moyenne des signaux disponibles."""
        values = [
            v for v in [
                self.momentum_composite,
                self.value,
                self.quality,
                self.low_volatility,
                self.mirofish_sentiment,
            ]
            if v is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)


def _fmt(v: float | None) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.3f}"


# ---------------------------------------------------------------------------
# Agrégateur principal
# ---------------------------------------------------------------------------


class SignalAggregator:
    """
    Calcule le vecteur de signal complet pour un symbole.

    Étapes :
    1. Facteurs quantitatifs (momentum, value, quality, vol)
    2. Enrichissement externe optionnel (MiroFish, QuantAgent, macro)
    3. Retourne un SignalVector normalisé et auditable

    Exemple :
        from data.client import OpenBBClient
        from signals.aggregator import SignalAggregator

        client = OpenBBClient()
        prices = client.ohlcv("AAPL", start="2022-01-01")

        agg = SignalAggregator()
        vector = agg.compute(prices, symbol="AAPL")
        print(vector.to_prompt())
    """

    def __init__(
        self,
        min_bars: int = 252,  # minimum 1 an de données
    ):
        self.min_bars = min_bars

        # Instanciation des facteurs
        self._factors = {
            "momentum_3m": MomentumFactor(lookback=63, skip_days=5),
            "momentum_12m": MomentumFactor(lookback=252, skip_days=21),
            "risk_adj_momentum": RiskAdjustedMomentum(),
            "trend_strength": TrendStrength(),
            "value": ValueFactor(),
            "quality": QualityFactor(),
            "low_volatility": VolatilityFactor(),
        }

    def compute(
        self,
        prices: pd.DataFrame,
        symbol: str = "UNKNOWN",
        macro: pd.DataFrame | None = None,
        mirofish_sentiment: float | None = None,
        agent_bias: float | None = None,
    ) -> SignalVector:
        """
        Args:
            prices             : DataFrame OHLCV (index DatetimeIndex)
            symbol             : ticker
            macro              : snapshot macro (output de client.macro_dashboard())
            mirofish_sentiment : score MiroFish (-1 → +1)
            agent_bias         : biais QuantAgent (-1 → +1)

        Returns:
            SignalVector avec toutes les valeurs disponibles
        """
        n_bars = len(prices.dropna(subset=["close"]))
        data_quality = min(1.0, n_bars / self.min_bars)

        if n_bars < 30:
            logger.error(f"{symbol}: seulement {n_bars} barres — signal impossible")
            return SignalVector(symbol=symbol, n_bars=n_bars, data_quality=0.0)

        if n_bars < self.min_bars:
            logger.warning(f"{symbol}: {n_bars} barres < {self.min_bars} minimum — signal dégradé")

        vector = SignalVector(
            symbol=symbol,
            n_bars=n_bars,
            data_quality=data_quality,
            mirofish_sentiment=mirofish_sentiment,
            agent_bias=agent_bias,
        )

        # Calcul des facteurs individuels
        for factor_name, factor in self._factors.items():
            try:
                raw = factor.compute(prices)
                last_val = raw.dropna().iloc[-1] if not raw.dropna().empty else None
                if last_val is not None and pd.notna(last_val):
                    setattr(vector, factor_name, round(float(last_val), 4))
            except Exception as e:
                logger.warning(f"{symbol} / {factor_name}: {e}")

        # Momentum composite (combinaison multi-horizon)
        try:
            comp = composite_momentum(prices)
            last = comp.dropna().iloc[-1]
            if pd.notna(last):
                vector.momentum_composite = round(float(last), 4)
        except Exception as e:
            logger.warning(f"{symbol} / composite_momentum: {e}")

        # Enrichissement macro
        if macro is not None:
            try:
                vector.vix = float(macro["vix"].iloc[-1]) if "vix" in macro.columns else None
                vector.spread_10y2y = float(macro["spread_10y2y"].iloc[-1]) if "spread_10y2y" in macro.columns else None
            except Exception as e:
                logger.warning(f"Macro enrichment: {e}")

        logger.info(
            f"{symbol} → composite={vector.composite_score:.3f} "
            f"quality={data_quality:.2f} bars={n_bars}"
            if vector.composite_score is not None
            else f"{symbol} → signal vide"
        )
        return vector

    def compute_multi(
        self,
        prices_dict: dict[str, pd.DataFrame],
        **kwargs,
    ) -> dict[str, SignalVector]:
        """Calcule le vecteur pour plusieurs symboles."""
        return {
            symbol: self.compute(prices, symbol=symbol, **kwargs)
            for symbol, prices in prices_dict.items()
        }

    def rank_universe(
        self,
        vectors: dict[str, SignalVector],
    ) -> pd.DataFrame:
        """
        Classe un univers de symboles par score composite.
        Utile pour la sélection de portefeuille.

        Returns:
            DataFrame trié par composite_score décroissant
        """
        rows = []
        for symbol, v in vectors.items():
            rows.append({
                "symbol": symbol,
                "composite": v.composite_score,
                "momentum": v.momentum_composite,
                "value": v.value,
                "quality": v.quality,
                "low_vol": v.low_volatility,
                "data_quality": v.data_quality,
            })
        df = pd.DataFrame(rows).set_index("symbol")
        return df.sort_values("composite", ascending=False)
