import pytest
import pandas as pd
import numpy as np
from signals.factors.risk_metrics import SkewnessFactor, KurtosisFactor, TailRatioFactor
from signals.aggregator import SignalAggregator

@pytest.fixture
def sample_prices():
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    # Générer des rendements avec une légère asymétrie et kurtosis
    returns = np.random.normal(0.001, 0.02, 500)
    # Ajouter quelques outliers pour le tail ratio
    returns[100] = 0.15
    returns[200] = -0.10

    close = 100 * (1 + returns).cumprod()
    df = pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "open": close,
        "volume": 1000
    }, index=dates)
    return df

def test_skewness_factor(sample_prices):
    factor = SkewnessFactor(window=252)
    signal = factor.compute(sample_prices)
    assert isinstance(signal, pd.Series)
    assert not signal.dropna().empty
    assert len(signal) == len(sample_prices)

def test_kurtosis_factor(sample_prices):
    factor = KurtosisFactor(window=252)
    signal = factor.compute(sample_prices)
    assert isinstance(signal, pd.Series)
    assert not signal.dropna().empty
    assert len(signal) == len(sample_prices)

def test_tail_ratio_factor(sample_prices):
    factor = TailRatioFactor(window=252)
    signal = factor.compute(sample_prices)
    assert isinstance(signal, pd.Series)
    assert not signal.dropna().empty
    assert len(signal) == len(sample_prices)
    # Le tail ratio doit être positif
    assert (signal.dropna() >= 0).all()

def test_aggregator_with_risk_metrics(sample_prices):
    agg = SignalAggregator()
    vector = agg.compute(sample_prices, symbol="TEST")

    assert vector.skewness is not None
    assert vector.kurtosis is not None
    assert vector.tail_ratio is not None

    prompt = vector.to_prompt()
    assert "skewness" in prompt
    assert "kurtosis" in prompt
    assert "tail_ratio" in prompt
