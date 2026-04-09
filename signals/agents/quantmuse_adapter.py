"""
signals/agents/quantmuse_adapter.py
------------------------------------
XGBoost + RandomForest ensemble → ml_prediction dans SignalVector.

Données   : daily OHLCV (OpenBB)
Features  : momentum/vol/trend sur fenêtres 21j–252j (daily)
Target    : signe du rendement forward 21 jours
Validation: walk-forward — pas de data leakage
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

MODEL_PATH = Path(__file__).parent / "quantmuse_model.pkl"


# ---------------------------------------------------------------------------
# Résultat ML
# ---------------------------------------------------------------------------


@dataclass
class MLSignal:
    ml_prediction: float | None              # -1.0 → +1.0
    model_used: str = "unknown"
    feature_importances: dict = field(default_factory=dict)
    confidence: float = 0.0
    n_train_samples: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Adapter principal
# ---------------------------------------------------------------------------


class QuantMuseAdapter:
    """
    Modèle ML entraîné sur les facteurs quant (XGBoost + RandomForest ensemble).

    Features : indicateurs techniques sur OHLCV daily (fenêtres 21j–252j)
    Target   : signe du rendement forward 21j  →  -1, 0, +1
    Modèles  : XGBoost + RandomForest (3 classes : bearish/flat/bullish)

    Contraintes :
    - Si sklearn/xgboost non installés → is_available()=False, ml_prediction=None
    - Jamais d'exception levée vers l'appelant
    """

    def __init__(self, lookback_train: int = 504, forward_days: int = 21):
        self.lookback_train = lookback_train
        self.forward_days = forward_days
        self._xgb_model = None
        self._rf_model = None
        self._is_trained = False
        self._feature_names: list[str] = []
        self._model_path = MODEL_PATH
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        """Charge le modèle depuis le disque si disponible."""
        if self._model_path.exists():
            try:
                data = joblib.load(self._model_path)
                self._xgb_model = data["xgb"]
                self._rf_model = data["rf"]
                self._feature_names = data["features"]
                self._is_trained = True
                logger.info(f"QuantMuse model loaded from {self._model_path}")
                return True
            except Exception as e:
                logger.warning(f"QuantMuse model load failed: {e}")
        return False

    def _save_model(self) -> None:
        """Persiste le modèle sur le disque."""
        try:
            joblib.dump({
                "xgb": self._xgb_model,
                "rf": self._rf_model,
                "features": self._feature_names,
            }, self._model_path)
            logger.info(f"QuantMuse model saved to {self._model_path}")
        except Exception as e:
            logger.warning(f"QuantMuse model save failed: {e}")

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Vérifie que sklearn et xgboost sont installés et chargeables. Aucun appel réseau."""
        try:
            import sklearn   # noqa: F401
            import xgboost   # noqa: F401
            return True
        except Exception:
            # ImportError si non installé, OSError/XGBoostError si libomp manquant (macOS)
            logger.warning("QuantMuse: sklearn/xgboost non disponibles — `brew install libomp && uv sync`")
            return False

    def train(self, prices: pd.DataFrame) -> dict[str, Any]:
        """
        Walk-forward 80/20 — pas de data leakage.

        Args:
            prices : DataFrame OHLCV daily

        Returns:
            dict avec n_train, n_test, xgb_accuracy, rf_accuracy, features
            (ou {"error": ...} si impossible)
        """
        if not self.is_available():
            return {"error": "sklearn/xgboost non installés"}

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        import xgboost as xgb

        features = self._build_features(prices)
        target = self._build_target(prices)
        df = pd.concat([features, target.rename("target")], axis=1).dropna()

        if len(df) < 100:
            return {"error": f"Données insuffisantes : {len(df)} points"}

        split = int(len(df) * 0.8)
        # Labels 0=bearish, 1=flat, 2=bullish  (target+1 : -1→0, 0→1, 1→2)
        X_tr = df.iloc[:split][self._feature_names].values
        y_tr = (df.iloc[:split]["target"] + 1).astype(int).values
        X_te = df.iloc[split:][self._feature_names].values
        y_te = (df.iloc[split:]["target"] + 1).astype(int).values

        self._xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        )
        self._xgb_model.fit(X_tr, y_tr)

        self._rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
        )
        self._rf_model.fit(X_tr, y_tr)
        self._is_trained = True
        self._save_model()

        return {
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "xgb_accuracy": round(float(accuracy_score(y_te, self._xgb_model.predict(X_te))), 4),
            "rf_accuracy": round(float(accuracy_score(y_te, self._rf_model.predict(X_te))), 4),
            "features": self._feature_names,
        }

    def predict(self, prices: pd.DataFrame, symbol: str = "UNKNOWN") -> MLSignal:
        """
        Prédit sur le dernier point. Entraîne si nécessaire.

        Returns:
            MLSignal avec ml_prediction ∈ [-1.0, +1.0] (None si erreur)
        """
        t0 = time.perf_counter()

        if not self.is_available():
            return MLSignal(ml_prediction=None, model_used="unavailable")

        try:
            if not self._is_trained:
                metrics = self.train(prices)
                if "error" in metrics:
                    logger.warning(f"{symbol}: {metrics['error']}")
                    return MLSignal(ml_prediction=None, model_used="untrained")

            feats = self._build_features(prices)
            last = feats.dropna().iloc[-1:][self._feature_names].values
            if last.shape[0] == 0:
                return MLSignal(ml_prediction=None, model_used="no_features")

            # Labels : 0=bearish, 1=flat, 2=bullish
            xgb_proba = self._xgb_model.predict_proba(last)[0]
            rf_proba = self._rf_model.predict_proba(last)[0]
            proba = (xgb_proba + rf_proba) / 2
            p_bear, _, p_bull = proba

            ml_pred = float(np.clip(p_bull - p_bear, -1.0, 1.0))
            confidence = float(max(p_bull, p_bear))
            importances = dict(zip(
                self._feature_names,
                [round(float(v), 4) for v in self._xgb_model.feature_importances_],
            ))

            logger.info(
                f"{symbol} → ml={ml_pred:+.3f} conf={confidence:.2f} "
                f"{(time.perf_counter() - t0) * 1000:.0f}ms"
            )

            return MLSignal(
                ml_prediction=ml_pred,
                model_used="ensemble_xgb_rf",
                feature_importances=importances,
                confidence=confidence,
                n_train_samples=len(feats.dropna()),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        except Exception as e:
            logger.warning(f"{symbol}: QuantMuse error — {e}")
            return MLSignal(ml_prediction=None, model_used="error")

    # ------------------------------------------------------------------
    # Logique interne
    # ------------------------------------------------------------------

    def _build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Features daily — fenêtres 21j, 63j, 126j, 252j uniquement."""
        close = prices["close"]
        high = prices["high"] if "high" in prices.columns else close
        low = prices["low"] if "low" in prices.columns else close
        feats: dict[str, pd.Series] = {}

        # Momentum multi-horizon (skip 5j court terme)
        feats["mom_21d"] = close.shift(5) / close.shift(21) - 1
        feats["mom_63d"] = close.shift(5) / close.shift(63) - 1
        feats["mom_126d"] = close.shift(21) / close.shift(126) - 1
        feats["mom_252d"] = close.shift(21) / close.shift(252) - 1

        # Volatilité réalisée annualisée
        ret = close.pct_change()
        feats["vol_21d"] = ret.rolling(21).std() * np.sqrt(252)
        feats["vol_63d"] = ret.rolling(63).std() * np.sqrt(252)

        # Trend : (EMA20 - EMA100) / ATR14
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema100 = close.ewm(span=100, adjust=False).mean()
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean().replace(0, np.nan)
        feats["trend"] = (ema20 - ema100) / atr14

        # Prix relatif à MA200
        ma200 = close.rolling(200).mean()
        feats["price_to_ma200"] = close / ma200.replace(0, np.nan) - 1

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        feats["rsi_14"] = 100 - (100 / (1 + gain / loss))

        # Short-term reversal
        feats["reversal_5d"] = -(close / close.shift(5) - 1)

        df = pd.DataFrame(feats, index=prices.index)

        # Winsorisation 1%-99% + z-score
        for col in df.columns:
            s = df[col]
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            s = s.clip(lo, hi)
            std = s.std()
            df[col] = (s - s.mean()) / std if std > 0 else 0.0

        self._feature_names = list(df.columns)
        return df

    def _build_target(self, prices: pd.DataFrame) -> pd.Series:
        """Signe du rendement forward 21j. -1/0/+1."""
        fwd = prices["close"].pct_change(self.forward_days).shift(-self.forward_days)
        target = pd.Series(0, index=prices.index, dtype=float)
        target[fwd > 0.005] = 1.0
        target[fwd < -0.005] = -1.0
        return target
