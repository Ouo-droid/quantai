import json
import os
import sys
import time

sys.path.insert(0, '.')
from dotenv import load_dotenv

load_dotenv()

from data.client import OpenBBClient
from execution.decision_agent import DecisionAgent
from signals.aggregator import SignalAggregator

print("=" * 60)
print("  TEST 1 — DecisionAgent RÉEL (claude-sonnet-4-6)")
print("=" * 60)

assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY manquante dans .env"
print("✓ ANTHROPIC_API_KEY présente")

client = OpenBBClient()
agg    = SignalAggregator()
agent  = DecisionAgent()

# Tracking usage
total_input_tokens = 0
total_output_tokens = 0
num_calls = 0

original_create = agent._client.messages.create
def tracked_create(*args, **kwargs):
    global total_input_tokens, total_output_tokens, num_calls
    res = original_create(*args, **kwargs)
    num_calls += 1
    total_input_tokens += res.usage.input_tokens
    total_output_tokens += res.usage.output_tokens
    return res

agent._client.messages.create = tracked_create

symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM"]

for sym in symbols:
    print(f"\n--- {sym} ---")
    try:
        # Signal complet
        prices = client.ohlcv(sym, start="2022-01-01")
        vector = agg.compute(prices, symbol=sym, use_quantmuse=True)
        print(f"  SignalVector : composite={vector.composite_score:+.3f} ml={vector.ml_prediction}")
        # print(f"  Prompt envoyé à Claude :\n{vector.to_prompt()}")

        # Vrai appel Claude
        t0 = time.perf_counter()
        order = agent.decide(vector)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n  → Claude répond en {elapsed:.0f}ms")
        print(f"  Direction   : {order.direction}")
        print(f"  Confidence  : {order.confidence:.2f}")
        print(f"  Stop Loss   : {order.stop_loss:.1%}")
        print(f"  Take Profit : {order.take_profit:.1%}")
        print(f"  Size        : {order.size_pct:.1%}")
        print(f"  Rationale   : {getattr(order, 'rationale', 'N/A')}")

        # Vérifications
        assert order.direction in ("LONG", "SHORT", "FLAT")
        assert 0.0 <= order.confidence <= 1.0
        assert 0.0 <= order.size_pct <= 0.10
        print("  ✓ Format ordre valide")

    except Exception as e:
        print(f"  ✗ ERREUR : {e}")

print("\nUsage Stats:")
print(f"Calls: {num_calls}")
print(f"Input tokens: {total_input_tokens}")
print(f"Output tokens: {total_output_tokens}")

# Save stats for later
with open('test_usage.json', 'w') as f:
    json.dump({
        "num_calls": num_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens
    }, f)
