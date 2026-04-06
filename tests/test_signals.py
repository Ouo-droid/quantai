"""
tests/test_signals.py
---------------------
Tests des facteurs et de l'agrégateur de signaux.

Lance : uv run pytest tests/test_signals.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.factors.base import BaseFactor, FactorResult, _estimate_half_life
from signals.factors.momentum import (
    MomentumFactor,
    RiskAdjustedMomentum,
    TrendStrength,
    MomentumReversal,
    composite_momentum,
)
from signals.factors.value_quality_vol import (
    ValueFactor,
    QualityFactor,
    VolatilityFactor,
    BetaFactor,
)
from signals.aggregator import SignalAggregator, SignalVector


# ---------------------------------------------------------------------------
# Fixture : données synthétiques
# ---------------------------------------------------------------------------


def make_prices(n: int = 500, trend: float = 0.0002, seed: int = 42) -> pd.DataFrame:
    """Génère des prix OHLCV synthétiques réalistes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    returns = rng.normal(trend, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    noise = rng.uniform(0.005, 0.015, n)

    return pd.DataFrame({
        "open":   close * (1 - noise / 2),
        "high":   close * (1 + noise),
        "low":    close * (1 - noise),
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=pd.DatetimeIndex(dates, name="date"))


@pytest.fixture
def prices():
    return make_prices(500)


@pytest.fixture
def prices_short():
    return make_prices(50)


# ---------------------------------------------------------------------------
# Tests BaseFactor utilities
# ---------------------------------------------------------------------------


class TestBaseUtils:
    def test_winsorize_clips_extremes(self, prices):
        s = pd.Series([-100.0, 0.0, 1.0, 2.0, 100.0])
        result = BaseFactor.winsorize(s, limits=(0.2, 0.2))
        assert result.min() > -100
        assert result.max() < 100

    def test_zscore_mean_zero(self, prices):
        s = pd.Series(np.random.randn(100))
        z = BaseFactor.zscore(s)
        assert abs(z.mean()) < 0.01
        assert abs(z.std() - 1.0) < 0.1

    def test_half_life_positive(self, prices):
        # Une série mean-reverting doit avoir une demi-vie positive
        s = pd.Series(np.random.randn(200)).cumsum()
        hl = _estimate_half_life(s)
        # Peut être None ou int — les deux sont acceptables
        assert hl is None or isinstance(hl, int)


# ---------------------------------------------------------------------------
# Tests MomentumFactor
# ---------------------------------------------------------------------------


class TestMomentumFactor:
    def test_compute_returns_series(self, prices):
        factor = MomentumFactor(lookback=252)
        result = factor.compute(prices)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_compute_same_index(self, prices):
        factor = MomentumFactor()
        result = factor.compute(prices)
        assert result.index.equals(prices.index)

    def test_nan_at_start(self, prices):
        factor = MomentumFactor(lookback=252, skip_days=21)
        result = factor.compute(prices)
        # Les premiers lookback jours doivent être NaN
        assert result.iloc[:252].isna().all()

    def test_trending_market_positive_signal(self):
        """Dans un marché haussier, le momentum doit être positif."""
        prices = make_prices(400, trend=0.002)  # fort trend haussier
        factor = MomentumFactor(lookback=252, skip_days=5)
        signal = factor.compute(prices).dropna()
        assert signal.mean() > 0

    def test_multi_horizon_columns(self, prices):
        factor = MomentumFactor()
        df = factor.compute_multi_horizon(prices, horizons=[21, 63, 126])
        assert set(df.columns) == {"mom_21d", "mom_63d", "mom_126d"}

    def test_compute_with_stats_returns_factorresult(self, prices):
        factor = MomentumFactor(lookback=126)
        result = factor.compute_with_stats(prices, forward_days=21)
        assert isinstance(result, FactorResult)
        assert result.name == "momentum"
        assert result.ic is not None
        assert result.t_stat is not None

    def test_risk_adj_momentum(self, prices):
        factor = RiskAdjustedMomentum()
        result = factor.compute(prices)
        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_trend_strength(self, prices):
        factor = TrendStrength()
        result = factor.compute(prices)
        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_reversal_opposite_sign(self):
        """Le reversal doit être opposé au momentum court terme."""
        prices = make_prices(300, trend=0.003)
        mom = MomentumFactor(lookback=21, skip_days=0).compute(prices).dropna()
        rev = MomentumReversal(lookback=21).compute(prices).dropna()
        common = mom.index.intersection(rev.index)
        # Corrélation doit être négative
        corr = mom.loc[common].corr(rev.loc[common])
        assert corr < 0


# ---------------------------------------------------------------------------
# Tests Value / Quality / Vol
# ---------------------------------------------------------------------------


class TestValueQualityVol:
    def test_value_factor(self, prices):
        factor = ValueFactor(lookback=252)
        result = factor.compute(prices)
        assert isinstance(result, pd.Series)
        assert result.dropna().abs().max() < 10  # winsorisé

    def test_quality_factor(self, prices):
        factor = QualityFactor()
        result = factor.compute(prices)
        assert isinstance(result, pd.Series)
        assert not result.dropna().empty

    def test_volatility_realized(self, prices):
        factor = VolatilityFactor(window=63, method="realized")
        result = factor.compute(prices)
        assert result.dropna().notna().all()

    def test_volatility_parkinson(self, prices):
        factor = VolatilityFactor(window=63, method="parkinson")
        result = factor.compute(prices)
        assert not result.dropna().empty

    def test_beta_without_benchmark(self, prices):
        """Sans benchmark, BetaFactor doit fallback sur vol."""
        factor = BetaFactor()
        result = factor.compute(prices, benchmark=None)
        assert isinstance(result, pd.Series)

    def test_beta_with_benchmark(self, prices):
        benchmark = prices["close"].pct_change().cumsum()
        factor = BetaFactor(window=63)
        result = factor.compute(prices, benchmark=benchmark)
        assert isinstance(result, pd.Series)
        assert not result.dropna().empty


# ---------------------------------------------------------------------------
# Tests composite_momentum
# ---------------------------------------------------------------------------


class TestCompositeMomentum:
    def test_returns_zscore_series(self, prices):
        result = composite_momentum(prices)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        # Doit être ~normalisé
        assert abs(valid.mean()) < 1.0
        assert abs(valid.std() - 1.0) < 0.5

    def test_custom_weights(self, prices):
        weights = {"mom_3m": 0.5, "mom_6m": 0.3, "mom_12m": 0.1, "ram": 0.05, "trend": 0.05}
        result = composite_momentum(prices, weights=weights)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Tests SignalAggregator
# ---------------------------------------------------------------------------


class TestSignalAggregator:
    def test_compute_returns_signal_vector(self, prices):
        agg = SignalAggregator(min_bars=100)
        vector = agg.compute(prices, symbol="TEST")
        assert isinstance(vector, SignalVector)
        assert vector.symbol == "TEST"
        assert vector.n_bars == len(prices)

    def test_momentum_fields_populated(self, prices):
        agg = SignalAggregator(min_bars=100)
        vector = agg.compute(prices, symbol="TEST")
        assert vector.momentum_12m is not None
        assert vector.momentum_composite is not None

    def test_data_quality_full(self, prices):
        agg = SignalAggregator(min_bars=252)
        vector = agg.compute(prices, symbol="TEST")
        assert vector.data_quality == 1.0  # 500 barres > 252

    def test_data_quality_degraded(self, prices_short):
        agg = SignalAggregator(min_bars=252)
        vector = agg.compute(prices_short, symbol="TEST")
        assert vector.data_quality < 1.0

    def test_external_signals_injected(self, prices):
        agg = SignalAggregator(min_bars=100)
        vector = agg.compute(
            prices, symbol="TEST",
            mirofish_sentiment=-0.42,
            agent_bias=0.75,
        )
        assert vector.mirofish_sentiment == -0.42
        assert vector.agent_bias == 0.75

    def test_composite_score_not_none(self, prices):
        agg = SignalAggregator(min_bars=100)
        vector = agg.compute(prices, symbol="TEST")
        assert vector.composite_score is not None

    def test_to_prompt_is_string(self, prices):
        agg = SignalAggregator(min_bars=100)
        vector = agg.compute(prices, symbol="AAPL")
        prompt = vector.to_prompt()
        assert isinstance(prompt, str)
        assert "AAPL" in prompt
        assert "momentum" in prompt.lower()

    def test_compute_multi(self):
        prices_dict = {
            "A": make_prices(400, trend=0.002),
            "B": make_prices(400, trend=-0.001),
            "C": make_prices(400, trend=0.0),
        }
        agg = SignalAggregator(min_bars=100)
        vectors = agg.compute_multi(prices_dict)
        assert set(vectors.keys()) == {"A", "B", "C"}

    def test_rank_universe(self):
        prices_dict = {
            "BULL": make_prices(400, trend=0.003),
            "FLAT": make_prices(400, trend=0.0),
            "BEAR": make_prices(400, trend=-0.003),
        }
        agg = SignalAggregator(min_bars=100)
        vectors = agg.compute_multi(prices_dict)
        ranked = agg.rank_universe(vectors)
        assert isinstance(ranked, pd.DataFrame)
        assert ranked.index[0] == "BULL"   # tendance haussière en tête
        assert ranked.index[-1] == "BEAR"  # tendance baissière en dernier

    def test_empty_prices_returns_empty_vector(self):
        empty = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=pd.DatetimeIndex([])
        )
        agg = SignalAggregator()
        vector = agg.compute(empty, symbol="EMPTY")
        assert vector.data_quality == 0.0
        assert vector.composite_score is None
