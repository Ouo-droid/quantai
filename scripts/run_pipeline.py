#!/usr/bin/env python
"""
Exécute le pipeline QuantAI complet sur un symbole.

Usage :
    python scripts/run_pipeline.py AAPL
    python scripts/run_pipeline.py BTC-USD --capital 50000
    python scripts/run_pipeline.py AAPL --quantagent
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.client import OpenBBClient
from execution.decision_agent import DecisionAgent
from execution.risk import Portfolio
from signals.aggregator import SignalAggregator


def run(symbol: str, capital: float = 100_000.0, use_quantagent: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"  QuantAI Pipeline — {symbol}")
    print(f"{'=' * 60}\n")

    # 1. Data
    print("[1/3] Fetching data from OpenBB...")
    client = OpenBBClient()
    prices = client.ohlcv(symbol, start="2024-01-01")
    print(f"      {len(prices)} bars loaded\n")

    # 2. Signal
    print("[2/3] Computing signal vector...")
    agg = SignalAggregator()
    vector = agg.compute(prices, symbol=symbol, use_quantagent=use_quantagent)
    print(vector.to_prompt())
    print()

    # 3. Décision + Risk
    print("[3/3] Decision Agent + Risk Engine...")
    portfolio = Portfolio(cash=capital)
    agent = DecisionAgent()

    # Compute recent returns for VaR check
    recent_returns: list[float] | None = None
    if "close" in prices.columns and len(prices) >= 20:
        recent_returns = prices["close"].pct_change().dropna().tolist()[-252:]

    order = agent.decide_with_risk(vector, portfolio, recent_returns=recent_returns)

    print()
    print(f"{'─' * 60}")
    if order is None:
        print("  → ORDRE BLOQUÉ par le Risk Engine")
    else:
        print(f"  → {order.direction}")
        print(f"     confidence : {order.confidence:.2f}")
        print(f"     size       : {order.size_pct:.1%} du portefeuille")
        print(f"     entry      : {order.entry:+.2%} vs marché")
        print(f"     stop_loss  : {order.stop_loss:.2%}")
        print(f"     take_profit: {order.take_profit:.2%}")
        print(f"     rationale  : {order.rationale}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantAI full pipeline")
    parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL, BTC-USD)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Portfolio capital in USD")
    parser.add_argument("--quantagent", action="store_true", help="Enable QuantAgent enrichment")
    args = parser.parse_args()

    run(args.symbol, args.capital, args.quantagent)
