"""
signals/factors/momentum.py
---------------------------
Facteurs momentum : cross-sectionnel et time-series.

Trois variantes :
  - MomentumFactor       : momentum brut (return N mois, skip 1 mois)
  - RiskAdjustedMomentum : momentum / volatilité réalisée (Sharpe-like)
  - TrendStrength        : ADX-inspired — force de la tendance

Référence académique :
  Jegadeesh & Titman (1993), Moskowitz, Ooi & Pedersen (2012)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFactor


class MomentumFactor(BaseFactor):
    """
    Momentum cross-sectionnel classique.

    Signal = rendement cumulé sur `lookback` jours,
    en excluant le dernier mois (skip 1 mois) pour éviter
    le mean-reversion court terme.

    Exemple :
        mom = MomentumFactor(lookback=252)
        signal = mom.compute(df)
        result = mom.compute_with_stats(df, forward_days=21)
        print(result.summary())
    """

    name = "momentum"

    def __init__(
        self,
        lookback: int = 252,    # ~12 mois
        skip_days: int = 21,    # skip dernier mois
        winsorize: bool = True,
    ):
        self.lookback = lookback
        self.skip_days = skip_days
        self._winsorize = winsorize

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Returns:
            pd.Series : rendement cumulé (lookback → skip), winsorisé
        """
        close = prices["close"]

        # Return de t-lookback à t-skip
        past_return = (
            close.shift(self.skip_days) / close.shift(self.lookback) - 1
        )

        if self._winsorize:
            past_return = self.winsorize(past_return)

        past_return.name = f"momentum_{self.lookback}d"
        return past_return

    def compute_multi_horizon(
        self,
        prices: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Calcule le momentum sur plusieurs horizons simultanément.

        Returns:
            DataFrame avec colonnes mom_21d, mom_63d, mom_126d, mom_252d
        """
        if horizons is None:
            horizons = [21, 63, 126, 252]

        results = {}
        for h in horizons:
            factor = MomentumFactor(lookback=h, skip_days=min(5, h // 10))
            results[f"mom_{h}d"] = factor.compute(prices)

        return pd.DataFrame(results)


class RiskAdjustedMomentum(BaseFactor):
    """
    Momentum ajusté par le risque (TSMOM de Moskowitz et al.).

    Signal = rendement_12m / volatilité_réalisée_1m
    → Similaire à un Sharpe ratio lookback

    Avantages vs momentum brut :
    - Moins sensible aux gros mouvements ponctuels
    - Naturellement normalisé entre actifs
    """

    name = "risk_adj_momentum"

    def __init__(
        self,
        return_window: int = 252,
        vol_window: int = 21,
        skip_days: int = 21,
    ):
        self.return_window = return_window
        self.vol_window = vol_window
        self.skip_days = skip_days

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change()

        # Rendement cumulé (skip inclus)
        past_ret = (
            close.shift(self.skip_days) / close.shift(self.return_window) - 1
        )

        # Volatilité réalisée annualisée
        # Floor à 1% annualisé : évite l'explosion du signal sur données quasi-constantes
        # (floating-point noise peut donner vol=1e-15 au lieu de 0 exact)
        MIN_VOL_ANNUALIZED = 0.01
        realized_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        realized_vol = realized_vol.clip(lower=MIN_VOL_ANNUALIZED)

        signal = past_ret / realized_vol
        signal = self.winsorize(signal)
        signal.name = "risk_adj_momentum"
        return signal


class TrendStrength(BaseFactor):
    """
    Force de tendance inspirée de l'ADX (Average Directional Index).

    Signal = |EMA_fast - EMA_slow| / ATR
    → Positif si tendance haussière, négatif si baissière,
      proche de 0 si en range.

    Utile comme filtre : ne trader le momentum que si
    TrendStrength > threshold.
    """

    name = "trend_strength"

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 100,
        atr_window: int = 14,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.atr_window = atr_window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        close = prices["close"]
        high = prices.get("high", close)
        low = prices.get("low", close)

        # EMAs
        ema_fast = close.ewm(span=self.fast_window, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_window, adjust=False).mean()

        # ATR (Average True Range)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_window, adjust=False).mean()
        atr = atr.replace(0, np.nan)

        # Signal normalisé
        signal = (ema_fast - ema_slow) / atr
        signal = self.winsorize(signal)
        signal.name = "trend_strength"
        return signal


class MomentumReversal(BaseFactor):
    """
    Short-term reversal (Jegadeesh 1990).

    Signal = -rendement_1m
    (les perdants court terme surperforment les gagnants)

    À combiner avec le momentum long terme pour filtrer
    les points d'entrée.
    """

    name = "st_reversal"

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        close = prices["close"]
        signal = -(close / close.shift(self.lookback) - 1)
        signal = self.winsorize(signal)
        signal.name = "st_reversal"
        return signal


# ---------------------------------------------------------------------------
# Combinaison de signaux momentum
# ---------------------------------------------------------------------------

def composite_momentum(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Signal momentum composite : moyenne pondérée de plusieurs horizons
    et du signal ajusté au risque.

    Args:
        prices  : DataFrame OHLCV
        weights : poids par signal (défaut : égaux)

    Returns:
        pd.Series normalisée (z-score)

    Exemple :
        signal = composite_momentum(df)
        # → signal prêt à entrer dans le vecteur de décision
    """
    factors = {
        "mom_3m":  MomentumFactor(lookback=63,  skip_days=5),
        "mom_6m":  MomentumFactor(lookback=126, skip_days=21),
        "mom_12m": MomentumFactor(lookback=252, skip_days=21),
        "ram":     RiskAdjustedMomentum(),
        "trend":   TrendStrength(),
    }

    if weights is None:
        weights = {k: 1.0 / len(factors) for k in factors}

    signals = {}
    for name, factor in factors.items():
        try:
            raw = factor.compute(prices)
            zscored = BaseFactor.zscore(raw)
            # Skip all-NaN signals (e.g. RAM/Trend on constant-price data)
            if zscored.dropna().empty:
                logger.warning(f"composite_momentum: {name} all-NaN — skipping")
                continue
            signals[name] = zscored
        except Exception as e:
            logger.warning(f"composite_momentum: {name} failed — {e}")

    if not signals:
        raise ValueError("Aucun signal calculable")

    df = pd.DataFrame(signals)
    # Renormalize weights to available signals so they always sum to 1.0
    total_weight = sum(weights.get(k, 0) for k in df.columns)
    if total_weight == 0:
        raise ValueError("Somme des poids = 0")
    composite = sum(df[k] * (weights.get(k, 0) / total_weight) for k in df.columns)
    composite = BaseFactor.zscore(composite)
    composite.name = "composite_momentum"
    return composite
