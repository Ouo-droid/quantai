"""
signals/factors/risk_metrics.py
------------------------------
Facteurs de moments d'ordre supérieur et métriques de risque.
Inspiré par l'étude Quantopian (2016) : "All that Glitters Is Not Gold".

L'étude montre que le Skewness, le Kurtosis et le Tail Ratio sont des
prédicteurs significatifs de la performance hors-échantillon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from .base import BaseFactor


class SkewnessFactor(BaseFactor):
    """
    Skewness (asymétrie) des rendements.
    Un skewness positif indique une distribution avec une queue à droite plus longue.
    """
    name = "skewness"

    def __init__(self, window: int = 252):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        returns = prices["close"].pct_change()
        signal = returns.rolling(self.window).skew()
        signal = self.winsorize(signal)
        signal.name = "skewness"
        return signal


class KurtosisFactor(BaseFactor):
    """
    Kurtosis (acuité) des rendements.
    Un kurtosis élevé indique des queues de distribution plus épaisses (fat tails),
    donc un risque de queue plus important.
    """
    name = "kurtosis"

    def __init__(self, window: int = 252):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        returns = prices["close"].pct_change()
        signal = returns.rolling(self.window).kurt()
        signal = self.winsorize(signal)
        signal.name = "kurtosis"
        return signal


class TailRatioFactor(BaseFactor):
    """
    Tail Ratio = abs(95th percentile) / abs(5th percentile).
    Indique si les gains extrêmes (queue droite) compensent les pertes extrêmes (queue gauche).
    Un ratio > 1 est généralement considéré comme favorable.
    """
    name = "tail_ratio"

    def __init__(self, window: int = 252):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        returns = prices["close"].pct_change()

        def _tail_ratio(x):
            # x is a numpy array when raw=True
            x = x[~np.isnan(x)]
            if len(x) < self.window // 2:
                return np.nan
            p95 = np.percentile(x, 95)
            p5 = np.percentile(x, 5)
            if abs(p5) < 1e-10:
                return 0.0
            return abs(p95) / abs(p5)

        signal = returns.rolling(self.window).apply(_tail_ratio, raw=True)
        signal = self.winsorize(signal)
        signal.name = "tail_ratio"
        return signal
