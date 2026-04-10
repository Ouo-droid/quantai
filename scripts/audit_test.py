"""
Script to audit the GitHub Research Agent functionality with Mock LLM.
Tests various financial scenarios and evaluates repo discovery.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from loguru import logger

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock LLM response content
class MockResponse:
    def __init__(self, content):
        self.content = content

def mock_prices(days=60):
    """Generates a mock DataFrame for testing."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    df = pd.DataFrame({
        'open': np.random.randn(days).cumsum() + 100,
        'high': np.random.randn(days).cumsum() + 105,
        'low': np.random.randn(days).cumsum() + 95,
        'close': np.random.randn(days).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, days)
    }, index=dates)
    return df

def audit_scenario(scenario_name, query, adapter):
    logger.info(f"--- Auditing Scenario: {scenario_name} ---")

    # We will manually call the github_agent_node to bypass the whole graph if needed,
    # but let's try to inject the mock into the adapter's run analysis if possible.
    # For simplicity in this audit script, let's test the github_agent directly.

    from signals.agents.quantagent.github_agent import create_github_agent

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MockResponse(query)

    github_node = create_github_agent(mock_llm)

    state = {
        "indicator_report": f"Report for {scenario_name}",
        "pattern_report": "Bullish",
        "trend_report": "Upward",
        "stock_name": scenario_name,
        "messages": []
    }

    try:
        result = github_node(state)

        github_report = result.get("github_report", "")
        github_repos = result.get("github_repos", [])

        if github_report:
            print(f"\n[GITHUB REPORT for {scenario_name}]")
            print(github_report)
        else:
            print(f"\n[!] No GitHub report generated for {scenario_name}")

        if github_repos:
            print(f"Discovered {len(github_repos)} repositories.")
            for repo in github_repos:
                print(f"- {repo['name']} (⭐ {repo['stars']})")
        else:
            print(f"[!] No GitHub repositories list found for {scenario_name}")

    except Exception as e:
        logger.error(f"Audit failed for {scenario_name}: {e}")

def main():
    # Mock adapter just to check availability
    adapter = MagicMock()
    adapter.is_available.return_value = True

    scenarios = [
        ("Portfolio Optimization", "portfolio optimization"),
        ("Risk Management VaR", "Value at Risk"),
        ("High Frequency Trading", "HFT trading strategy"),
        ("Financial Audit & Compliance", "financial audit compliance"),
        ("Empty Query Test", "finance"),
    ]

    for name, query in scenarios:
        audit_scenario(name, query, adapter)

if __name__ == "__main__":
    main()
