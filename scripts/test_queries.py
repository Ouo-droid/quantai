
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.decision_agent import DecisionAgent
from signals.aggregator import SignalVector
from datetime import datetime, UTC

def test_query(query):
    print(f"\nTesting query: {query}")
    agent = DecisionAgent()
    # Create a dummy signal vector where we put the query in the symbol or rationale
    vector = SignalVector(
        symbol=query,
        timestamp=datetime.now(UTC),
        momentum_composite=0.5,
        value=0.2,
        quality=0.8,
        agent_bias=0.6
    )

    try:
        order = agent.decide(vector)
        print(f"Decision: {order.direction}")
        print(f"Rationale: {order.rationale}")
        print(f"Active: {order.is_active()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    queries = [
        "Analyse de risque VaR",
        "Algorithme de trading momentum",
        "Reporting financier ESG",
        "Audit de portefeuille"
    ]
    for q in queries:
        test_query(q)
