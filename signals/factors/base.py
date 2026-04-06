"""
signals/factors/base.py
-----------------------
Classe de base pour tous les facteurs quantitatifs.
Chaque facteur hérite de BaseFactor et implémente compute().

Convention : compute() retourne toujours une pd.Series
avec le même index que les prix en entrée.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class FactorResult:
    """Résultat d'un facteur : valeur + statistiques de validité."""
    name: str
    values: pd.Series                  # signal brut
    values_ranked: pd.Series | None = None   # rang cross-sectionnel 0→1
    ic: float | None = None            # information coefficient (corr vs fwd return)
    t_stat: float | None = None        # t-statistique du IC
    p_value: float | None = None
    half_life_days: int | None = None  # demi-vie du signal

    @property
    def is_significant(self) -> bool:
        """p-value < 5% ET |t-stat| > 2."""
        if self.p_value is None or self.t_stat is None:
            return False
        return self.p_value < 0.05 and abs(self.t_stat) > 2.0

    def summary(self) -> dict[str, Any]:
        return {
            "factor": self.name,
            "ic": round(self.ic, 4) if self.ic else None,
            "t_stat": round(self.t_stat, 3) if self.t_stat else None,
            "p_value": round(self.p_value, 4) if self.p_value else None,
            "significant": self.is_significant,
            "half_life_days": self.half_life_days,
        }


class BaseFactor(ABC):
    """
    Interface commune pour tous les facteurs.

    Sous-classes : MomentumFactor, ValueFactor, QualityFactor, VolatilityFactor
    """

    name: str = "base"

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calcule le signal brut.

        Args:
            prices : DataFrame OHLCV avec index DatetimeIndex
            **kwargs : paramètres spécifiques au facteur

        Returns:
            pd.Series avec même index que prices, valeurs numériques
        """
        ...

    def compute_with_stats(
        self,
        prices: pd.DataFrame,
        forward_days: int = 21,
        **kwargs,
    ) -> FactorResult:
        """
        Calcule le facteur + IC vs rendement forward + t-stat.
        Utile pour la validation du signal avant deployment.
        """
        signal = self.compute(prices, **kwargs)

        # Rendement forward
        fwd_return = prices["close"].pct_change(forward_days).shift(-forward_days)

        # Aligner
        common = signal.dropna().index.intersection(fwd_return.dropna().index)
        if len(common) < 30:
            logger.warning(f"{self.name}: pas assez de données pour IC ({len(common)} points)")
            return FactorResult(name=self.name, values=signal)

        s = signal.loc[common]
        f = fwd_return.loc[common]

        # IC = corrélation rang (Spearman)
        ic, p_value = stats.spearmanr(s, f)
        t_stat = ic * np.sqrt((len(common) - 2) / (1 - ic**2 + 1e-10))

        # Rang cross-sectionnel 0→1
        ranked = signal.rank(pct=True)

        # Demi-vie : autocorrélation AR(1)
        half_life = _estimate_half_life(signal.dropna())

        return FactorResult(
            name=self.name,
            values=signal,
            values_ranked=ranked,
            ic=float(ic),
            t_stat=float(t_stat),
            p_value=float(p_value),
            half_life_days=half_life,
        )

    @staticmethod
    def winsorize(s: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
        """Clip les extrêmes pour éviter l'influence des outliers."""
        lo = s.quantile(limits[0])
        hi = s.quantile(1 - limits[1])
        return s.clip(lower=lo, upper=hi)

    @staticmethod
    def zscore(s: pd.Series) -> pd.Series:
        """Normalisation z-score (mean=0, std=1)."""
        return (s - s.mean()) / (s.std() + 1e-10)


def _estimate_half_life(series: pd.Series) -> int | None:
    """
    Estime la demi-vie d'un signal via régression AR(1).
    half_life = -log(2) / log(beta) où beta = coeff AR(1)
    """
    try:
        s = series.dropna()
        if len(s) < 20:
            return None
        y = s.diff().dropna()
        x = s.shift(1).dropna()
        common = y.index.intersection(x.index)
        slope, _, _, _, _ = stats.linregress(x.loc[common], y.loc[common])
        if slope >= 0:
            return None
        hl = int(-np.log(2) / np.log(1 + slope))
        return max(1, min(hl, 1000))  # clamp raisonnable
    except Exception:
        return None
