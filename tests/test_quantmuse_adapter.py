"""
tests/test_quantmuse_adapter.py
--------------------------------
Tests unitaires du QuantMuseAdapter et de l'intégration SignalAggregator.

Aucune dépendance réseau — données synthétiques uniquement.

Lance : uv run pytest tests/test_quantmuse_adapter.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from signals.agents.quantmuse_adapter import MLSignal, QuantMuseAdapter
from signals.aggregator import SignalAggregator, SignalVector

# ---------------------------------------------------------------------------
# Fixture : prix synthétiques (500 barres daily, format OpenBB)
# ---------------------------------------------------------------------------


def make_prices(n: int = 500, trend: float = 0.0002, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    returns = rng.normal(trend, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    noise = rng.uniform(0.005, 0.015, n)
    return pd.DataFrame(
        {
            "open":   close * (1 - noise / 2),
            "high":   close * (1 + noise),
            "low":    close * (1 - noise),
            "close":  close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture()
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture()
def adapter() -> QuantMuseAdapter:
    return QuantMuseAdapter()


# ---------------------------------------------------------------------------
# 1. is_available
# ---------------------------------------------------------------------------


def test_is_available_returns_bool(adapter):
    result = adapter.is_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2. SignalVector a le champ ml_prediction
# ---------------------------------------------------------------------------


def test_signal_vector_has_ml_prediction_field():
    v = SignalVector(symbol="TEST")
    assert hasattr(v, "ml_prediction")
    assert v.ml_prediction is None


# ---------------------------------------------------------------------------
# 3. Features — fenêtres daily présentes
# ---------------------------------------------------------------------------


def test_build_features_daily_windows(adapter, prices):
    feats = adapter._build_features(prices)
    for col in ("mom_21d", "mom_63d", "vol_21d"):
        assert col in feats.columns, f"colonne manquante : {col}"


# ---------------------------------------------------------------------------
# 4. Features — aucune colonne intraday / HFT
# ---------------------------------------------------------------------------


def test_build_features_no_intraday_columns(adapter, prices):
    feats = adapter._build_features(prices)
    intraday_markers = ("1min", "5min", "hft", "tick", "1s", "15min")
    for col in feats.columns:
        for marker in intraday_markers:
            assert marker not in col.lower(), (
                f"Colonne intraday détectée : {col!r} contient {marker!r}"
            )


# ---------------------------------------------------------------------------
# 5. Target ∈ {-1.0, 0.0, +1.0}
# ---------------------------------------------------------------------------


def test_build_target_values(adapter, prices):
    target = adapter._build_target(prices)
    unique = set(target.dropna().unique())
    assert unique <= {-1.0, 0.0, 1.0}, f"Valeurs inattendues : {unique}"


# ---------------------------------------------------------------------------
# 6. Walk-forward 80/20 — n_train > n_test
# ---------------------------------------------------------------------------


def test_train_walk_forward_split(adapter, prices):
    if not adapter.is_available():
        pytest.skip("sklearn/xgboost non installés")
    metrics = adapter.train(prices)
    assert "error" not in metrics
    assert metrics["n_train"] > metrics["n_test"]


# ---------------------------------------------------------------------------
# 7. train() retourne xgb_accuracy et rf_accuracy
# ---------------------------------------------------------------------------


def test_train_returns_accuracy(adapter, prices):
    if not adapter.is_available():
        pytest.skip("sklearn/xgboost non installés")
    metrics = adapter.train(prices)
    assert "error" not in metrics
    assert "xgb_accuracy" in metrics
    assert "rf_accuracy" in metrics
    assert 0.0 <= metrics["xgb_accuracy"] <= 1.0
    assert 0.0 <= metrics["rf_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# 8. ml_prediction ∈ [-1.0, +1.0]
# ---------------------------------------------------------------------------


def test_predict_ml_prediction_in_range(adapter, prices):
    if not adapter.is_available():
        pytest.skip("sklearn/xgboost non installés")
    signal = adapter.predict(prices, symbol="TEST")
    if signal.ml_prediction is not None:
        assert -1.0 <= signal.ml_prediction <= 1.0


# ---------------------------------------------------------------------------
# 9. predict() ne lève jamais d'exception
# ---------------------------------------------------------------------------


def test_predict_never_raises(adapter, prices):
    # Corruption volontaire des données
    bad_prices = prices.copy()
    bad_prices["close"] = np.nan
    try:
        result = adapter.predict(bad_prices, symbol="BAD")
    except Exception as exc:
        pytest.fail(f"predict() a levé une exception : {exc}")
    assert isinstance(result, MLSignal)


# ---------------------------------------------------------------------------
# 10. predict() sur DataFrame vide → ml_prediction=None
# ---------------------------------------------------------------------------


def test_predict_empty_df(adapter):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = adapter.predict(empty, symbol="EMPTY")
    assert result.ml_prediction is None


# ---------------------------------------------------------------------------
# 11. SignalAggregator — use_quantmuse=False par défaut → ml_prediction=None
# ---------------------------------------------------------------------------


def test_aggregator_skips_quantmuse_by_default():
    agg = SignalAggregator()
    p = make_prices(300)
    vector = agg.compute(p, symbol="DEFAULT")
    assert vector.ml_prediction is None


# ---------------------------------------------------------------------------
# 12. SignalAggregator — use_quantmuse=True remplit ml_prediction
# ---------------------------------------------------------------------------


def test_aggregator_uses_quantmuse_when_requested():
    import signals.agents.quantmuse_adapter as _qm_mod

    mock_adapter = MagicMock()
    mock_adapter.is_available.return_value = True
    mock_adapter.predict.return_value = MLSignal(
        ml_prediction=0.42,
        model_used="ensemble_xgb_rf",
        confidence=0.7,
    )
    mock_cls = MagicMock(return_value=mock_adapter)

    original = _qm_mod.QuantMuseAdapter
    _qm_mod.QuantMuseAdapter = mock_cls  # type: ignore
    try:
        agg = SignalAggregator()
        vector = agg.compute(make_prices(300), symbol="AAPL", use_quantmuse=True)
    finally:
        _qm_mod.QuantMuseAdapter = original

    mock_adapter.predict.assert_called_once()
    assert vector.ml_prediction == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# 13. to_prompt() inclut ml_prediction
# ---------------------------------------------------------------------------


def test_ml_prediction_in_to_prompt():
    v = SignalVector(symbol="X", ml_prediction=0.342)
    prompt = v.to_prompt()
    assert "ml_prediction" in prompt
    assert "+0.342" in prompt


# ---------------------------------------------------------------------------
# 14. confidence ∈ [0, 1]
# ---------------------------------------------------------------------------


def test_confidence_in_01(adapter, prices):
    if not adapter.is_available():
        pytest.skip("sklearn/xgboost non installés")
    signal = adapter.predict(prices, symbol="TEST")
    assert 0.0 <= signal.confidence <= 1.0


# ---------------------------------------------------------------------------
# 15. feature_importances non vide après predict
# ---------------------------------------------------------------------------


def test_feature_importances_not_empty(adapter, prices):
    if not adapter.is_available():
        pytest.skip("sklearn/xgboost non installés")
    signal = adapter.predict(prices, symbol="TEST")
    if signal.ml_prediction is not None:
        assert len(signal.feature_importances) > 0
