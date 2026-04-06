# CONTEXT — QuantAI session handoff
# Commit ce fichier : git add HANDOFF.md && git commit -m "docs: session handoff"

## Projet
Système de trading quantitatif de pointe combinant :
- OpenBB (data layer)
- QuantAgent (agents LLM visuels)
- MiroFish (simulation sociale / stress scenarios)
- QuantMuse (facteurs quant + C++ execution)

Repo GitHub : https://github.com/Ouo-droid/quantai

---

## État actuel — ce qui est codé et fonctionne

### ✅ Layer 0 — Data (OpenBB)
- `data/client.py` — wrapper OpenBB complet
  - `ohlcv(symbol, start, end)` → DataFrame OHLCV ✅ testé (566 barres AAPL)
  - `news(symbol)` → list[NewsItem] (provider yfinance, gratuit)
  - `macro_dashboard()` → VIX via yfinance + FRED si clé dispo
  - `fundamentals(symbol)` → P/E, P/B, ROE, D/E
  - `health()` → ping serveur
- `scripts/check_openbb.py` — script de vérification ✅ passé
- `tests/test_data.py` — 10 tests unitaires ✅ tous verts

### ✅ Layer 1 — Signals (facteurs quant)
- `signals/factors/base.py` — BaseFactor, FactorResult, half-life estimator
- `signals/factors/momentum.py`
  - MomentumFactor (lookback configurable, skip 1 mois)
  - RiskAdjustedMomentum (momentum / vol réalisée)
  - TrendStrength (EMA fast/slow / ATR)
  - MomentumReversal (short-term mean reversion)
  - composite_momentum() — z-score multi-horizon
- `signals/factors/value_quality_vol.py`
  - ValueFactor, FundamentalValueFactor
  - QualityFactor, FundamentalQualityFactor
  - VolatilityFactor (realized + Parkinson)
  - BetaFactor (low-beta anomaly)
- `signals/aggregator.py`
  - SignalVector — dataclass avec tous les champs (momentum, value, quality, vol, mirofish_sentiment, agent_bias, macro)
  - SignalAggregator.compute() → SignalVector
  - SignalAggregator.rank_universe() → DataFrame trié par score composite
  - SignalVector.to_prompt() → texte pour Decision Agent LLM
- `tests/test_signals.py` — 30 tests ✅ tous verts

### ✅ Infrastructure
- `pyproject.toml` — dépendances pinées (openbb, anthropic, langchain, zmq, streamlit...)
- `.github/workflows/ci.yml` — CI GitHub Actions (lint ruff + mypy + pytest)
- `.env.example` — template clés API
- `README.md` — structure documentée

---

## Ce qui manque (roadmap)

### 🔜 Prochaine étape — research/01_momentum_factor.ipynb
Notebook Jupyter de backtest momentum sur S&P 500 :
- Univers : S&P 500 (ou subset 50 stocks)
- Période : 2015–2025
- Signal : MomentumFactor(lookback=252) + composite_momentum()
- Métriques : Sharpe annualisé, max drawdown, hit rate, t-stat IC
- Visualisations : courbe de performance, distribution IC, turnover
- Objectif : notebook publiable qui donne de la crédibilité au repo GitHub

### 🔜 execution/risk.py
- RiskEngine avec limites VaR, CVaR, drawdown, position sizing
- Validation synchrone avant tout ordre

### 🔜 execution/decision_agent.py
- Decision Agent Claude (claude-sonnet-4-6)
- Input : SignalVector.to_prompt()
- Output : ordre JSON structuré {direction, confidence, entry, stop_loss, take_profit, size_pct}
- Règle hard : confidence < 0.6 → FLAT automatique

### 🔜 simulation/mirofish_client.py
- Client HTTP vers MiroFish (localhost:5001)
- Convertit news OpenBB → seed MiroFish
- Retourne sentiment_index → injecté dans SignalVector.mirofish_sentiment

### 🔜 dashboard/app.py
- Streamlit : positions live, P&L, signaux, scénarios MiroFish

---

## Stack technique
- Python 3.13, uv comme package manager
- OpenBB serveur sur localhost:6900 (lancer avec : openbb-api)
- Claude API : claude-sonnet-4-6
- ZMQ pour C++ ↔ Python (execution layer)

## Clés API nécessaires
- ANTHROPIC_API_KEY → requis pour Decision Agent
- FRED_API_KEY → gratuit sur fred.stlouisfed.org (macro CPI/taux)
- Benzinga → payant, remplacé par yfinance pour les news

## Commandes utiles
```bash
# Activer l'env
source .venv/bin/activate

# Lancer OpenBB
openbb-api

# Vérifier data layer
python scripts/check_openbb.py

# Tests
python -m pytest tests/ -v -k "not integration"

# Push
git add -A && git commit -m "..." && git push
```

---

## Prompt de reprise pour Claude

Voici le contexte exact pour reprendre :

> Je travaille sur un système de trading quantitatif appelé QuantAI
> (repo : https://github.com/Ouo-droid/quantai).
> 
> Ce qui est déjà codé et fonctionne :
> - data/client.py : wrapper OpenBB (OHLCV ✅, macro ✅, news ✅)
> - signals/factors/ : MomentumFactor, RiskAdjustedMomentum, TrendStrength,
>   ValueFactor, QualityFactor, VolatilityFactor, BetaFactor
> - signals/aggregator.py : SignalVector + SignalAggregator (40 tests verts)
> - CI GitHub Actions configurée
> 
> Prochaine étape : créer research/01_momentum_factor.ipynb
> Backtest momentum sur un univers d'actions (2015-2025),
> avec Sharpe, drawdown, IC t-stat — notebook propre et publiable
> pour donner de la crédibilité au repo GitHub.
> 
> Stack : Python 3.13, OpenBB sur localhost:6900, uv, claude-sonnet-4-6.
