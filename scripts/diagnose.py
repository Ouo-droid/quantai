"""
scripts/diagnose.py
--------------------
Diagnostic end-to-end de toutes les couches QuantAI.

Usage : uv run python scripts/diagnose.py
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SYMBOL = "AAPL"
START  = "2024-01-01"

results: list[dict] = []


def check(name: str, fn) -> bool:
    t0 = time.monotonic()
    try:
        detail = fn()
        ms = (time.monotonic() - t0) * 1000
        results.append({"name": name, "ok": True, "detail": detail, "ms": ms})
        console.print(f"  [green]✓[/green] {name} [dim]({ms:.0f}ms)[/dim]")
        if detail:
            console.print(f"    [dim]{detail}[/dim]")
        return True
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        results.append({"name": name, "ok": False, "detail": str(e), "ms": ms})
        console.print(f"  [red]✗[/red] {name}")
        console.print(f"    [red]{e}[/red]")
        return False


# =============================================================================
# LAYER 0 — OpenBB data
# =============================================================================

console.rule("[bold cyan]Layer 0 — OpenBB Data")

from data.client import OpenBBClient
client = OpenBBClient()

prices = None

def test_health():
    ok = client.health()
    if not ok:
        raise RuntimeError("OpenBB server ne répond pas")
    return "serveur joignable"

def test_ohlcv():
    global prices
    prices = client.ohlcv(SYMBOL, start=START)
    assert len(prices) > 100, f"seulement {len(prices)} barres"
    assert "close" in prices.columns
    return f"{len(prices)} barres | {prices.index[0].date()} → {prices.index[-1].date()}"

def test_ohlcv_columns():
    assert prices is not None
    missing = [c for c in ["open","high","low","close","volume"] if c not in prices.columns]
    if missing:
        raise AssertionError(f"Colonnes manquantes : {missing}")
    return f"colonnes OK : {list(prices.columns)}"

def test_news():
    items = client.news(SYMBOL, limit=5, provider="yfinance")
    if not items:
        return "0 news (yfinance peut être vide)"
    return f"{len(items)} news | ex: \"{items[0].title[:60]}...\""

def test_fundamentals():
    f = client.fundamentals(SYMBOL)
    return f"PE={f.pe_ratio} | PB={f.pb_ratio} | ROE={f.roe}"

check("health()", test_health)
check("ohlcv(AAPL)", test_ohlcv)
check("colonnes OHLCV", test_ohlcv_columns)
check("news(AAPL)", test_news)
check("fundamentals(AAPL)", test_fundamentals)


# =============================================================================
# LAYER 1 — Signals
# =============================================================================

console.rule("[bold cyan]Layer 1 — Signals")

import pandas as pd
from signals.factors.momentum import (
    MomentumFactor, RiskAdjustedMomentum, TrendStrength, composite_momentum
)
from signals.factors.value_quality_vol import (
    ValueFactor, QualityFactor, VolatilityFactor
)
from signals.aggregator import SignalAggregator, SignalVector

signal_vector: SignalVector | None = None

def test_momentum_factor():
    factor = MomentumFactor(lookback=63)
    series = factor.compute(prices)
    last = series.dropna().iloc[-1]
    return f"momentum_3m={last:.4f}"

def test_risk_adj_momentum():
    factor = RiskAdjustedMomentum()
    series = factor.compute(prices)
    last = series.dropna().iloc[-1]
    return f"risk_adj={last:.4f}"

def test_trend_strength():
    factor = TrendStrength()
    series = factor.compute(prices)
    last = series.dropna().iloc[-1]
    return f"trend={last:.4f}"

def test_value_factor():
    factor = ValueFactor()
    series = factor.compute(prices)
    last = series.dropna().iloc[-1]
    return f"value={last:.4f}"

def test_volatility_factor():
    factor = VolatilityFactor()
    series = factor.compute(prices)
    last = series.dropna().iloc[-1]
    return f"vol={last:.4f}"

def test_composite_momentum():
    series = composite_momentum(prices)
    last = series.dropna().iloc[-1]
    return f"composite_mom={last:.4f}"

def test_signal_aggregator():
    global signal_vector
    agg = SignalAggregator()
    signal_vector = agg.compute(prices, symbol=SYMBOL)
    assert signal_vector.symbol == SYMBOL
    assert signal_vector.n_bars > 0
    assert signal_vector.composite_score is not None
    return (
        f"n_bars={signal_vector.n_bars} | "
        f"composite={signal_vector.composite_score:.4f} | "
        f"data_quality={signal_vector.data_quality:.2f}"
    )

def test_to_prompt():
    assert signal_vector is not None
    prompt = signal_vector.to_prompt()
    assert "MOMENTUM" in prompt
    assert SYMBOL in prompt
    return f"{len(prompt)} chars générés"

check("MomentumFactor(63j)", test_momentum_factor)
check("RiskAdjustedMomentum", test_risk_adj_momentum)
check("TrendStrength", test_trend_strength)
check("ValueFactor", test_value_factor)
check("VolatilityFactor", test_volatility_factor)
check("composite_momentum()", test_composite_momentum)
check("SignalAggregator.compute()", test_signal_aggregator)
check("SignalVector.to_prompt()", test_to_prompt)


# =============================================================================
# LAYER 1b — QuantAgent adapter (sans appel LLM)
# =============================================================================

console.rule("[bold cyan]Layer 1b — QuantAgent Adapter")

from signals.agents.quantagent_adapter import QuantAgentAdapter

adapter = QuantAgentAdapter(llm_provider="anthropic")

def test_quantagent_is_available():
    avail = adapter.is_available()
    return f"is_available={avail}"

def test_quantagent_to_kline():
    kline = adapter._to_kline_dict(prices)
    keys = list(kline.keys())
    lengths = {k: len(v) for k, v in kline.items()}
    assert len(kline["Open"]) == 30, f"attendu 30, got {len(kline['Open'])}"
    return f"keys={keys} | 30 lignes chacune"

def test_quantagent_parse_long():
    import json
    raw = json.dumps({
        "decision": "LONG",
        "forecast_horizon": "next 3 candles",
        "justification": "bullish momentum",
        "risk_reward_ratio": 1.6
    })
    direction, bias, conf = QuantAgentAdapter._parse_decision(raw)
    assert direction == "LONG", f"direction={direction}"
    assert bias is not None and 0.3 <= bias <= 1.0
    return f"LONG → bias={bias:.3f} confidence={conf:.3f}"

def test_quantagent_parse_short():
    import json
    raw = json.dumps({
        "decision": "SHORT",
        "justification": "bearish",
        "risk_reward_ratio": 1.4
    })
    direction, bias, conf = QuantAgentAdapter._parse_decision(raw)
    assert direction == "SHORT"
    assert bias is not None and -1.0 <= bias <= -0.3
    return f"SHORT → bias={bias:.3f}"

def test_quantagent_graceful_fail():
    # Simule un timeout / LLM down → ne doit pas lever
    from unittest.mock import patch
    with patch.object(adapter, "_run_analysis", side_effect=Exception("LLM down")):
        signal = adapter.analyze(prices, symbol=SYMBOL)
    assert signal.agent_bias is None
    assert signal.direction == "UNKNOWN"
    return "Exception absorbée → agent_bias=None ✓"

def test_aggregator_with_quantagent_flag():
    """use_quantagent=True sans clé API → agent_bias reste None, pas de crash."""
    from unittest.mock import patch, MagicMock
    import signals.agents.quantagent_adapter as _mod
    mock_adapter = MagicMock()
    mock_adapter.is_available.return_value = False  # simule clé absente
    original = _mod.QuantAgentAdapter
    _mod.QuantAgentAdapter = MagicMock(return_value=mock_adapter)
    try:
        agg = SignalAggregator()
        vec = agg.compute(prices, symbol=SYMBOL, use_quantagent=True)
    finally:
        _mod.QuantAgentAdapter = original
    assert vec.agent_bias is None
    return "pipeline OK sans QuantAgent (is_available=False)"

check("QuantAgentAdapter.is_available()", test_quantagent_is_available)
check("_to_kline_dict() → 30 barres", test_quantagent_to_kline)
check("parse_decision(LONG)", test_quantagent_parse_long)
check("parse_decision(SHORT)", test_quantagent_parse_short)
check("graceful_fail (LLM down)", test_quantagent_graceful_fail)
check("aggregator use_quantagent=True sans LLM", test_aggregator_with_quantagent_flag)


# =============================================================================
# LAYER 1c — QuantAgent submodule integrity
# =============================================================================

console.rule("[bold cyan]Layer 1c — QuantAgent Submodule")

from pathlib import Path
QA_PATH = Path("signals/agents/quantagent")

def test_submodule_present():
    assert QA_PATH.exists(), "dossier absent"
    assert (QA_PATH / "trading_graph.py").exists()
    assert (QA_PATH / "agent_state.py").exists()
    assert (QA_PATH / "graph_setup.py").exists()
    files = [f.name for f in QA_PATH.iterdir() if f.suffix == ".py"]
    return f"{len(files)} fichiers .py: {files}"

def test_submodule_importable():
    """Import des modules QuantAgent sans initialiser le LLM."""
    import sys
    if str(QA_PATH.resolve()) not in sys.path:
        sys.path.insert(0, str(QA_PATH.resolve()))
    # Import des modules stateless uniquement
    import agent_state  # noqa
    import default_config  # noqa
    return "agent_state + default_config importés"

def test_kline_state_fields():
    import agent_state
    fields = list(agent_state.IndicatorAgentState.__annotations__.keys())
    required = ["kline_data", "time_frame", "stock_name",
                "indicator_report", "pattern_image", "trend_image",
                "final_trade_decision"]
    missing = [f for f in required if f not in fields]
    if missing:
        raise AssertionError(f"Champs manquants dans IndicatorAgentState: {missing}")
    return f"{len(fields)} champs présents"

check("submodule présent + fichiers clés", test_submodule_present)
check("import agent_state sans LLM", test_submodule_importable)
check("IndicatorAgentState champs requis", test_kline_state_fields)


# =============================================================================
# LAYER 2 — Decision Agent (Claude)
# =============================================================================

console.rule("[bold cyan]Layer 2 — Decision Agent (Claude)")

from execution.decision_agent import DecisionAgent, TradeOrder

agent = DecisionAgent()
trade_order: TradeOrder | None = None

def test_agent_init():
    assert agent is not None
    return "DecisionAgent instancié"

def test_parse_order_long():
    import json
    raw = json.dumps({
        "direction": "LONG",
        "confidence": 0.82,
        "entry": 185.0,
        "stop_loss": 182.0,
        "take_profit": 192.0,
        "size_pct": 0.03,
        "rationale": "momentum fort"
    })
    order = agent._parse_order(raw, SYMBOL)
    assert order.direction == "LONG"
    assert order.confidence == 0.82
    return f"LONG parsed → conf={order.confidence} entry={order.entry}"

def test_parse_order_low_confidence():
    import json
    raw = json.dumps({
        "direction": "LONG",
        "confidence": 0.45,  # < 0.6 → FLAT
        "entry": 185.0,
        "stop_loss": 183.0,
        "take_profit": 190.0,
        "size_pct": 0.02,
    })
    order = agent._parse_order(raw, SYMBOL)
    assert order.direction == "FLAT", f"attendu FLAT, got {order.direction}"
    return f"confidence=0.45 → FLAT ✓"

def test_live_decide():
    global trade_order
    assert signal_vector is not None

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Vérifie si la clé est accessible via le SDK (credentials internes)
        try:
            import anthropic
            anthropic.Anthropic().models.list()
            api_key = "found_via_sdk"
        except Exception:
            pass

    if not api_key:
        # Génère un ordre fictif pour ne pas bloquer le diagnostic
        trade_order = TradeOrder(
            symbol=SYMBOL, direction="FLAT", confidence=0.0,
            rationale="[SKIPPED] ANTHROPIC_API_KEY absent de l'env shell. "
                       "Crée un fichier .env avec ta clé, ou exporte : "
                       "export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return f"[SKIP] clé absente → crée .env à partir de .env.example"

    trade_order = agent.decide(signal_vector)
    assert isinstance(trade_order, TradeOrder)
    assert trade_order.direction in ("LONG", "SHORT", "FLAT")
    assert 0.0 <= trade_order.confidence <= 1.0
    return (
        f"direction={trade_order.direction} | "
        f"confidence={trade_order.confidence:.2f} | "
        f"size_pct={trade_order.size_pct:.3f}"
    )

check("DecisionAgent init", test_agent_init)
check("parse_order(LONG)", test_parse_order_long)
check("parse_order(low confidence → FLAT)", test_parse_order_low_confidence)
check("decide(SignalVector) [appel Claude API réel]", test_live_decide)


# =============================================================================
# RÉSUMÉ
# =============================================================================

console.rule("[bold white]Résumé")

table = Table(show_header=True, header_style="bold")
table.add_column("Check", style="white", min_width=40)
table.add_column("Status", min_width=8)
table.add_column("ms", justify="right")
table.add_column("Détail", style="dim", max_width=60)

n_ok = n_fail = 0
for r in results:
    status = "[green]✓ OK[/green]" if r["ok"] else "[red]✗ FAIL[/red]"
    ms = f"{r['ms']:.0f}"
    detail = str(r["detail"])[:60] if r["detail"] else ""
    table.add_row(r["name"], status, ms, detail)
    if r["ok"]:
        n_ok += 1
    else:
        n_fail += 1

console.print(table)

color = "green" if n_fail == 0 else "red"
console.print(Panel(
    f"[{color}]{n_ok}/{n_ok + n_fail} checks passés[/{color}]"
    + (f"\n[red]{n_fail} échec(s)[/red]" if n_fail else ""),
    title="QuantAI Diagnostic",
))

if trade_order and trade_order.direction != "FLAT":
    console.print(Panel(
        f"[bold]{SYMBOL}[/bold] → [yellow]{trade_order.direction}[/yellow]\n"
        f"Confidence : {trade_order.confidence:.0%}\n"
        f"Entry       : {trade_order.entry}\n"
        f"Stop Loss   : {trade_order.stop_loss}\n"
        f"Take Profit : {trade_order.take_profit}\n"
        f"Size        : {trade_order.size_pct:.1%} du portefeuille",
        title="Ordre Decision Agent",
        border_style="yellow",
    ))

sys.exit(0 if n_fail == 0 else 1)
