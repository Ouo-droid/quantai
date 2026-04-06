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

## Licence

MIT
