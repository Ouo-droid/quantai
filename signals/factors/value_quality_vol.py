"""
signals/factors/value_quality_vol.py
-------------------------------------
Facteurs value, quality et volatilité.

Value     : prix relatif à la valeur fondamentale
Quality   : profitabilité et solidité du bilan
Volatility: vol réalisée, vol implicite, beta

Ces trois facteurs + momentum constituent le socle classique
des modèles multi-facteurs (Fama-French, AQR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFactor


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------


class ValueFactor(BaseFactor):
    """
    Facteur value basé sur les prix uniquement (pas de fondamentaux requis).

    Signal = -rendement_long_terme (prix "bas" = potentiel de revalorisation)
    Proxy de value sans données fondamentales — utile pour crypto/ETF.

    Pour un vrai facteur value actions, utiliser FundamentalValueFactor.
    """

    name = "value_price"

    def __init__(self, lookback: int = 756):  # ~3 ans
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        close = prices["close"]
        # Prix actuel vs moyenne long terme — déviation inversée
        long_ma = close.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        signal = -(close / long_ma - 1)  # bas vs moyenne = signal value positif
        signal = self.winsorize(signal)
        signal.name = "value_price"
        return signal


class FundamentalValueFactor(BaseFactor):
    """
    Facteur value fondamental — nécessite des données comptables.

    Signal composite de : P/E inverse, P/B inverse, rendement FCF.
    Les données fondamentales sont passées via kwargs.

    Exemple :
        from data.client import OpenBBClient
        client = OpenBBClient()
        fund = client.fundamentals("AAPL")
        signal = FundamentalValueFactor().compute(df, fundamentals=fund)
    """

    name = "value_fundamental"

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals=None,
        **kwargs,
    ) -> pd.Series:
        if fundamentals is None:
            logger.warning("FundamentalValueFactor: pas de fondamentaux, fallback sur ValueFactor")
            return ValueFactor().compute(prices)

        close = prices["close"]
        signals = []

        if fundamentals.pe_ratio and fundamentals.pe_ratio > 0:
            earnings_yield = pd.Series(1.0 / fundamentals.pe_ratio, index=close.index)
            signals.append(self.zscore(earnings_yield))

        if fundamentals.pb_ratio and fundamentals.pb_ratio > 0:
            book_yield = pd.Series(1.0 / fundamentals.pb_ratio, index=close.index)
            signals.append(self.zscore(book_yield))

        if not signals:
            return ValueFactor().compute(prices)

        composite = pd.concat(signals, axis=1).mean(axis=1)
        composite.name = "value_fundamental"
        return composite


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class QualityFactor(BaseFactor):
    """
    Facteur qualité basé sur les prix (proxy).

    Signal = stabilité des rendements + faible drawdown
    → Les entreprises "quality" ont des rendements plus stables

    Proxy sans fondamentaux : utile pour toutes les classes d'actifs.
    """

    name = "quality_stability"

    def __init__(
        self,
        vol_window: int = 252,
        drawdown_window: int = 252,
    ):
        self.vol_window = vol_window
        self.drawdown_window = drawdown_window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        close = prices["close"]
        returns = close.pct_change()

        # 1. Stabilité = inverse de la volatilité
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        stability = -vol  # moins volatile = plus quality

        # 2. Max drawdown sur la fenêtre
        rolling_max = close.rolling(self.drawdown_window).max()
        drawdown = (close - rolling_max) / rolling_max
        low_drawdown = -drawdown  # drawdown proche de 0 = quality

        # 3. Composite
        signal = (self.zscore(stability) + self.zscore(low_drawdown)) / 2
        signal = self.winsorize(signal)
        signal.name = "quality_stability"
        return signal


class FundamentalQualityFactor(BaseFactor):
    """
    Facteur qualité fondamental.

    Signal composite de : ROE, D/E inversé
    """

    name = "quality_fundamental"

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals=None,
        **kwargs,
    ) -> pd.Series:
        if fundamentals is None:
            logger.warning("FundamentalQualityFactor: fallback sur QualityFactor")
            return QualityFactor().compute(prices)

        close = prices["close"]
        signals = []

        if fundamentals.roe is not None:
            roe = pd.Series(fundamentals.roe, index=close.index)
            signals.append(self.zscore(roe))

        if fundamentals.debt_to_equity is not None and fundamentals.debt_to_equity >= 0:
            low_leverage = pd.Series(-fundamentals.debt_to_equity, index=close.index)
            signals.append(self.zscore(low_leverage))

        if not signals:
            return QualityFactor().compute(prices)

        composite = pd.concat(signals, axis=1).mean(axis=1)
        composite.name = "quality_fundamental"
        return composite


# ---------------------------------------------------------------------------
# Volatilité
# ---------------------------------------------------------------------------


class VolatilityFactor(BaseFactor):
    """
    Facteur volatilité (low-vol anomaly).

    Signal = -volatilité réalisée
    → Les actifs à faible vol surperforment sur le long terme
      (Blitz & van Vliet 2007, Baker et al. 2011)

    Trois métriques :
    - vol_realized  : std des rendements quotidiens
    - vol_parkinson : estimateur Parkinson (high-low) — plus efficace
    - vol_beta      : beta vs benchmark (à fournir)
    """

    name = "low_volatility"

    def __init__(
        self,
        window: int = 63,  # ~3 mois
        method: str = "realized",  # "realized" | "parkinson"
        annualize: bool = True,
    ):
        self.window = window
        self.method = method
        self.annualize = annualize

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if self.method == "parkinson" and "high" in prices.columns and "low" in prices.columns:
            signal = self._parkinson(prices)
        else:
            signal = self._realized(prices)

        # Low-vol anomaly : signal = -vol
        signal = -signal
        signal = self.winsorize(signal)
        signal.name = "low_volatility"
        return signal

    def _realized(self, prices: pd.DataFrame) -> pd.Series:
        returns = prices["close"].pct_change()
        vol = returns.rolling(self.window).std()
        if self.annualize:
            vol = vol * np.sqrt(252)
        return vol

    def _parkinson(self, prices: pd.DataFrame) -> pd.Series:
        """
        Estimateur Parkinson : utilise high et low plutôt que close.
        Variance estimée = (1/4ln2) * E[(ln(H/L))^2]
        ~5x plus efficace que la vol réalisée classique.
        """
        log_hl = np.log(prices["high"] / prices["low"])
        pk_var = (log_hl**2) / (4 * np.log(2))
        vol = pk_var.rolling(self.window).mean().apply(np.sqrt)
        if self.annualize:
            vol = vol * np.sqrt(252)
        return vol


class BetaFactor(BaseFactor):
    """
    Beta vs benchmark (low-beta anomaly).

    Signal = -beta
    → Les actions à faible beta délivrent un alpha ajusté au risque
      supérieur (Black 1972, Frazzini & Pedersen 2014 — BAB factor)
    """

    name = "low_beta"

    def __init__(self, window: int = 252):
        self.window = window

    def compute(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series | None = None,
        **kwargs,
    ) -> pd.Series:
        if benchmark is None:
            logger.warning("BetaFactor: pas de benchmark fourni, retourne zscore vol")
            return VolatilityFactor().compute(prices)

        stock_ret = prices["close"].pct_change()
        bench_ret = benchmark.pct_change()

        # Alignement
        common = stock_ret.dropna().index.intersection(bench_ret.dropna().index)
        s = stock_ret.loc[common]
        b = bench_ret.loc[common]

        # Beta rolling = cov(s,b) / var(b)
        cov = s.rolling(self.window).cov(b)
        var = b.rolling(self.window).var()
        beta = cov / (var + 1e-10)

        # Low-beta = signal positif
        signal = -beta.reindex(prices.index)
        signal = self.winsorize(signal)
        signal.name = "low_beta"
        return signal
