#!/usr/bin/env python
"""
Test MiroFish avec données réelles OpenBB.

Usage :
    # 1. Lancer MiroFish dans un terminal séparé
    # 2. python scripts/test_mirofish.py AAPL

Nécessite : openbb-api + MiroFish sur localhost:5001
"""
import argparse
import sys

sys.path.insert(0, ".")

from data.client import OpenBBClient
from simulation.mirofish_client import MiroFishClient


def main(symbol: str = "AAPL") -> None:
    print(f"\n=== MiroFish test — {symbol} ===")

    mf = MiroFishClient()
    if not mf.health():
        print("! MiroFish non disponible — lancer depuis github.com/666ghj/MiroFish")
        print("  En mode dégradé : mirofish_sentiment=None")
        return

    print("✓ MiroFish disponible")

    client = OpenBBClient()
    news = client.news(symbol, limit=10)
    print(f"✓ {len(news)} articles récupérés pour {symbol}")

    seed = MiroFishClient.news_to_seed(news)
    print(f"✓ Seed : {len(seed)} articles filtrés")
    for s in seed[:3]:
        print(f"  - {s['title'][:60]}")

    print(f"\n→ Simulation en cours ({mf.n_agents} agents, {mf.n_rounds} rounds)...")
    result = mf.simulate(seed, scenario="market_news", symbol=symbol)

    if result.sentiment_index is not None:
        print("\nRésultat :")
        print(f"  sentiment_index : {result.sentiment_index:+.3f}")
        print(f"  panic_spread    : {result.panic_spread:.3f}")
        print(f"  latency         : {result.latency_ms:.0f}ms")
    else:
        print("x Simulation échouée")

    print("\n→ Stress test : rate_hike_200bps")
    shock = mf.simulate_macro_shock("rate_hike_200bps", magnitude=1.5, symbol=symbol)
    if shock.sentiment_index is not None:
        print(f"  sentiment après choc : {shock.sentiment_index:+.3f}")
    else:
        print("  (simulation échouée)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="AAPL")
    args = parser.parse_args()
    main(args.symbol)
