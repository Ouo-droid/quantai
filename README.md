# QuantAI

> Système de trading quantitatif de pointe — OpenBB · QuantAgent · MiroFish · QuantMuse

[![CI](https://github.com/YOUR_USERNAME/quantai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/quantai/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Architecture

```
OpenBB (data)  →  Signals (facteurs + agents LLM)  →  Decision Agent
      ↓                        ↓                              ↓
  News / Macro          MiroFish (simulation)          Risk Engine
                                                            ↓
                                                    Execution (C++)
```

## Stack

| Couche | Technologie |
|--------|------------|
| Data | OpenBB Platform |
| Facteurs | QuantMuse (Python + C++) |
| Agents visuels | QuantAgent (LangGraph) |
| Simulation macro | MiroFish (multi-agent) |
| Décision | Claude (Anthropic API) |
| Exécution | C++ core via ZMQ |
| Dashboard | Streamlit + OpenBB Workspace |

## Démarrage rapide

```bash
# 1. Cloner
git clone https://github.com/YOUR_USERNAME/quantai
cd quantai

# 2. Installer (uv recommandé)
pip install uv
uv sync --extra dev

# 3. Config
cp .env.example .env
# Éditer .env avec tes clés API

# 4. Lancer OpenBB
openbb-api  # → http://127.0.0.1:6900

# 5. Vérifier
uv run python scripts/check_openbb.py

# 6. Tests
uv run pytest tests/ -v
```

## Structure du projet

```
quantai/
├── data/              # Client OpenBB unifié
├── signals/
│   ├── factors/       # Momentum, value, quality, vol
│   └── agents/        # QuantAgent (LangGraph)
├── simulation/        # MiroFish stress scenarios
├── execution/         # Risk engine + C++ router
├── dashboard/         # Streamlit
├── research/          # Notebooks reproductibles
└── tests/
```

## Research notebooks

| Notebook | Description |
|----------|-------------|
| `research/01_momentum_factor.ipynb` | Momentum cross-sectionnel — Sharpe, drawdown, t-stat |
| (WIP) `research/02_mirofish_macro.ipynb` | MiroFish comme générateur de stress scenarios |

## Agent cluster (QuantAgent)

QuantAgent est intégré comme submodule Git dans `signals/agents/quantagent/`.
Il fournit une analyse technique visuelle (indicateurs, patterns, tendances)
via des agents LangGraph avec vision LLM. Sa sortie (`agent_bias`) est injectée
dans le `SignalVector` avant la décision finale.

```bash
# Initialiser le submodule après un git clone
git submodule update --init --recursive

# Installer les dépendances agents
uv sync --extra agents
```

```python
from signals.agents.quantagent_adapter import QuantAgentAdapter

adapter = QuantAgentAdapter(llm_provider="anthropic")  # ou "openai", "qwen", "minimax"
if adapter.is_available():
    signal = adapter.analyze(prices_df, symbol="AAPL")
    print(signal.agent_bias)    # float -1.0 → +1.0
    print(signal.direction)     # "LONG" | "SHORT" | "FLAT" | "UNKNOWN"

# Intégration directe dans le pipeline de signaux
from signals.aggregator import SignalAggregator
agg = SignalAggregator()
vector = agg.compute(prices_df, symbol="AAPL", use_quantagent=True)
print(vector.agent_bias)
```

## Licence

MIT
