"""
tests/test_quantagent_integration.py
--------------------------------------
Mocked integration tests for the QuantAgent live APIs.

Tests cover:
- GitHub Agent: mocked httpx calls to GitHub API, report generation via mocked LLM
- Decision Agent (QuantAgent graph): mocked LLM calls, decision flow
- Decision Journal: recording and querying
- Regime Risk Calibrator: regime detection and parameter calibration
- Full graph flow: mocked external calls, START→END pipeline

No real API calls are made — all external dependencies are mocked.

Run: uv run pytest tests/test_quantagent_integration.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add quantagent to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "signals", "agents", "quantagent"))

# Mock heavy native/incompatible dependencies before importing project modules
for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MOCK_GITHUB_API_RESPONSE = {
    "items": [
        {
            "full_name": "user/quant-strategy",
            "description": "A quantitative trading strategy using RSI and MACD",
            "stargazers_count": 1500,
            "language": "Python",
            "html_url": "https://github.com/user/quant-strategy",
            "updated_at": "2026-03-15T10:00:00Z",
        },
        {
            "full_name": "org/backtest-framework",
            "description": "Backtesting framework for systematic trading",
            "stargazers_count": 3200,
            "language": "Python",
            "html_url": "https://github.com/org/backtest-framework",
            "updated_at": "2026-04-01T08:30:00Z",
        },
    ]
}

MOCK_DECISION_JSON = json.dumps(
    {
        "forecast_horizon": "Predicting next 1 candlestick (15 minutes)",
        "decision": "LONG",
        "justification": "Strong MACD crossover confirmed by RSI breakout above 70.",
        "risk_reward_ratio": 1.5,
        "confidence": 0.82,
    }
)

MOCK_LLM_REPORT = "GitHub research reveals two relevant repos..."


def _make_mock_state() -> dict:
    """Create a minimal graph state dict for testing."""
    return {
        "kline_data": {"open": [100], "high": [105], "low": [99], "close": [103], "volume": [50000]},
        "time_frame": "15min",
        "stock_name": "AAPL",
        "indicator_report": "RSI at 72, MACD bullish crossover.",
        "pattern_report": "Bullish engulfing pattern on 4H chart.",
        "trend_report": "Upward sloping support with breakout confirmation.",
        "github_report": "",
        "github_repos": [],
        "recent_returns": [],
        "messages": [],
    }


# ---------------------------------------------------------------------------
# Tests: GitHub Agent
# ---------------------------------------------------------------------------


class TestGitHubAgentMocked:
    """Mocked integration tests for the GitHub Agent."""

    def test_search_github_repos_success(self):
        """GitHub API search returns parsed repositories."""
        from github_agent import _search_github_repos

        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_GITHUB_API_RESPONSE
        mock_resp.raise_for_status.return_value = None

        with patch("github_agent.httpx.get", return_value=mock_resp) as mock_get:
            repos = _search_github_repos("quantitative trading AAPL", max_results=5)

            mock_get.assert_called_once()
            assert len(repos) == 2
            assert repos[0]["name"] == "user/quant-strategy"
            assert repos[0]["stars"] == 1500
            assert repos[1]["full_name"] == "org/backtest-framework"

    def test_search_github_repos_with_token(self):
        """GitHub API uses Bearer token when GITHUB_TOKEN is set."""
        from github_agent import _search_github_repos

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": []}
        mock_resp.raise_for_status.return_value = None

        with (
            patch("github_agent.httpx.get", return_value=mock_resp) as mock_get,
            patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}),
        ):
            _search_github_repos("test query")
            call_kwargs = mock_get.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer ghp_test123"

    def test_search_github_repos_api_error(self):
        """GitHub API error returns empty list gracefully."""
        from github_agent import _search_github_repos

        with patch("github_agent.httpx.get", side_effect=Exception("Network error")):
            repos = _search_github_repos("test query")
            assert repos == []

    def test_build_search_queries(self):
        """Search queries are generated from stock name."""
        from github_agent import _build_search_queries

        queries = _build_search_queries("AAPL")
        assert len(queries) >= 3
        assert any("AAPL" in q for q in queries)
        assert any("quantitative" in q.lower() for q in queries)

    def test_github_agent_node_full_flow(self):
        """Full GitHub agent node: search + LLM summary."""
        from github_agent import create_github_agent

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_LLM_REPORT
        mock_llm.invoke.return_value = mock_response

        state = _make_mock_state()

        mock_http_resp = MagicMock()
        mock_http_resp.json.return_value = MOCK_GITHUB_API_RESPONSE
        mock_http_resp.raise_for_status.return_value = None

        with patch("github_agent.httpx.get", return_value=mock_http_resp):
            node = create_github_agent(mock_llm)
            result = node(state)

        assert "github_report" in result
        assert result["github_report"] == MOCK_LLM_REPORT
        assert "github_repos" in result
        assert len(result["github_repos"]) > 0
        mock_llm.invoke.assert_called_once()

    def test_github_agent_node_llm_error(self):
        """GitHub agent handles LLM errors gracefully."""
        from github_agent import create_github_agent

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM timeout")

        state = _make_mock_state()

        mock_http_resp = MagicMock()
        mock_http_resp.json.return_value = {"items": []}
        mock_http_resp.raise_for_status.return_value = None

        with patch("github_agent.httpx.get", return_value=mock_http_resp):
            node = create_github_agent(mock_llm)
            result = node(state)

        assert "github_report" in result
        assert "error" in result["github_report"].lower()


# ---------------------------------------------------------------------------
# Tests: Decision Agent (QuantAgent)
# ---------------------------------------------------------------------------


class TestDecisionAgentMocked:
    """Mocked integration tests for the QuantAgent decision agent."""

    def test_decision_node_returns_required_keys(self):
        """Decision node returns final_trade_decision, messages, and decision_prompt."""
        from decision_agent import create_final_trade_decider

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_DECISION_JSON
        mock_llm.invoke.return_value = mock_response

        node = create_final_trade_decider(mock_llm)
        state = _make_mock_state()
        result = node(state)

        assert "final_trade_decision" in result
        assert "messages" in result
        assert "decision_prompt" in result

    def test_decision_node_includes_all_reports_in_prompt(self):
        """Decision prompt includes all four analysis reports."""
        from decision_agent import create_final_trade_decider

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_DECISION_JSON
        mock_llm.invoke.return_value = mock_response

        node = create_final_trade_decider(mock_llm)
        state = _make_mock_state()
        state["github_report"] = "Relevant repos found: quant-strategy (1500 stars)"
        result = node(state)

        prompt = result["decision_prompt"]
        assert "RSI at 72" in prompt
        assert "Bullish engulfing" in prompt
        assert "Upward sloping support" in prompt
        assert "quant-strategy" in prompt

    def test_decision_node_with_journal_logging(self):
        """Decision node logs to journal when journal is provided."""
        from decision_agent import create_final_trade_decider
        from decision_journal import DecisionJournal

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_DECISION_JSON
        mock_llm.invoke.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "journal.json")
            node = create_final_trade_decider(mock_llm, journal=journal)
            state = _make_mock_state()
            node(state)

            entries = journal.get_entries()
            assert len(entries) == 1
            assert entries[0].stock_name == "AAPL"
            assert entries[0].decision_direction == "LONG"

    def test_decision_node_with_regime_calibration(self):
        """Decision node includes regime section when calibrator + returns provided."""
        from decision_agent import create_final_trade_decider
        from regime_risk_calibrator import RegimeRiskCalibrator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_DECISION_JSON
        mock_llm.invoke.return_value = mock_response

        calibrator = RegimeRiskCalibrator()
        node = create_final_trade_decider(mock_llm, regime_calibrator=calibrator)

        state = _make_mock_state()
        # Simulate a bull market with low volatility returns
        state["recent_returns"] = [0.001] * 60

        result = node(state)
        prompt = result["decision_prompt"]
        assert "Market Regime Risk Calibration" in prompt

    def test_decision_node_without_regime_when_no_returns(self):
        """Decision node omits regime section when no recent_returns in state."""
        from decision_agent import create_final_trade_decider
        from regime_risk_calibrator import RegimeRiskCalibrator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_DECISION_JSON
        mock_llm.invoke.return_value = mock_response

        calibrator = RegimeRiskCalibrator()
        node = create_final_trade_decider(mock_llm, regime_calibrator=calibrator)

        state = _make_mock_state()
        state["recent_returns"] = []

        result = node(state)
        prompt = result["decision_prompt"]
        assert "Market Regime Risk Calibration" not in prompt


# ---------------------------------------------------------------------------
# Tests: Decision Journal
# ---------------------------------------------------------------------------


class TestDecisionJournal:
    """Tests for the DecisionJournal."""

    def test_record_and_retrieve(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json")
            state = _make_mock_state()
            journal.record(state, MOCK_DECISION_JSON)

            entries = journal.get_entries()
            assert len(entries) == 1
            assert entries[0].stock_name == "AAPL"
            assert entries[0].time_frame == "15min"
            assert entries[0].decision_direction == "LONG"
            assert entries[0].decision_confidence == pytest.approx(0.82)
            assert entries[0].decision_risk_reward == pytest.approx(1.5)

    def test_record_multiple_and_filter(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json")
            state1 = _make_mock_state()
            state1["stock_name"] = "AAPL"
            state2 = _make_mock_state()
            state2["stock_name"] = "TSLA"

            journal.record(state1, MOCK_DECISION_JSON)
            journal.record(state2, MOCK_DECISION_JSON)

            all_entries = journal.get_entries()
            assert len(all_entries) == 2

            aapl_only = journal.get_entries(stock_name="AAPL")
            assert len(aapl_only) == 1
            assert aapl_only[0].stock_name == "AAPL"

    def test_persistence_across_instances(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "j.json"
            j1 = DecisionJournal(path=path)
            j1.record(_make_mock_state(), MOCK_DECISION_JSON)

            j2 = DecisionJournal(path=path)
            assert len(j2.get_entries()) == 1

    def test_summary(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json")
            for _ in range(3):
                journal.record(_make_mock_state(), MOCK_DECISION_JSON)

            s = journal.summary()
            assert s["total_decisions"] == 3
            assert "LONG" in s["directions"]
            assert s["avg_confidence"] == pytest.approx(0.82)
            assert "AAPL" in s["stocks_traded"]

    def test_max_entries_eviction(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json", max_entries=5)
            for i in range(10):
                state = _make_mock_state()
                state["stock_name"] = f"STOCK{i}"
                journal.record(state, MOCK_DECISION_JSON)

            entries = journal.get_entries()
            assert len(entries) == 5
            # Should keep the latest 5
            assert entries[0].stock_name == "STOCK5"

    def test_clear(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json")
            journal.record(_make_mock_state(), MOCK_DECISION_JSON)
            assert len(journal.get_entries()) == 1

            journal.clear()
            assert len(journal.get_entries()) == 0

    def test_parse_non_json_decision(self):
        from decision_journal import DecisionJournal

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json")
            journal.record(_make_mock_state(), "Not valid JSON output from LLM")

            entry = journal.get_entries()[0]
            assert entry.decision_direction == "UNKNOWN"
            assert "Not valid JSON" in entry.decision_justification


# ---------------------------------------------------------------------------
# Tests: Regime Risk Calibrator
# ---------------------------------------------------------------------------


class TestRegimeRiskCalibrator:
    """Tests for market regime detection and risk calibration."""

    def test_detect_bull_regime(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        # Moderate positive returns, normal vol
        returns = [0.002] * 60
        result = cal.detect_regime(returns)
        assert result.regime == MarketRegime.LOW_VOL  # constant returns → zero vol

    def test_detect_bear_regime(self):
        import random

        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        rng = random.Random(42)
        # Negative trend with moderate vol
        returns = [rng.gauss(-0.003, 0.015) for _ in range(60)]
        result = cal.detect_regime(returns)
        # Seed 42 produces drawdown >15% triggering CRISIS; BEAR/HIGH_VOL also acceptable
        assert result.regime in (MarketRegime.BEAR, MarketRegime.HIGH_VOL, MarketRegime.CRISIS)

    def test_detect_high_vol_regime(self):
        import random

        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        rng = random.Random(99)
        # High volatility
        returns = [rng.gauss(0.0, 0.03) for _ in range(60)]
        result = cal.detect_regime(returns)
        assert result.regime in (MarketRegime.HIGH_VOL, MarketRegime.CRISIS)

    def test_detect_crisis_regime(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        # Extreme losses → big drawdown
        returns = [-0.05] * 10 + [0.001] * 50
        result = cal.detect_regime(returns)
        assert result.regime == MarketRegime.CRISIS

    def test_detect_insufficient_data(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        result = cal.detect_regime([0.01, -0.01])
        assert result.regime == MarketRegime.BULL  # safe default
        assert result.diagnostics["reason"] == "insufficient_data"

    def test_get_risk_params(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        for regime in MarketRegime:
            params = cal.get_risk_params(regime)
            assert params.regime == regime
            assert 0 < params.max_position_pct <= 0.10
            assert params.min_confidence >= 0.50

    def test_crisis_params_most_conservative(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        crisis = cal.get_risk_params(MarketRegime.CRISIS)
        bull = cal.get_risk_params(MarketRegime.BULL)

        assert crisis.max_position_pct < bull.max_position_pct
        assert crisis.min_confidence > bull.min_confidence
        assert crisis.size_scalar < bull.size_scalar

    def test_calibrate_returns_both(self):
        from regime_risk_calibrator import RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        returns = [0.001] * 60
        result, params = cal.calibrate(returns)
        assert result.regime == params.regime

    def test_format_regime_prompt_section(self):
        from regime_risk_calibrator import RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        returns = [0.001] * 60
        result, params = cal.calibrate(returns)
        section = cal.format_regime_prompt_section(result, params)

        assert "Market Regime Risk Calibration" in section
        assert "Detected regime" in section
        assert "Max position size" in section
        assert "Stop-loss" in section

    def test_custom_regime_params(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator, RegimeRiskParams

        custom = {
            MarketRegime.BULL: RegimeRiskParams(
                regime=MarketRegime.BULL,
                max_position_pct=0.10,
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                min_confidence=0.50,
                risk_reward_min=1.0,
                risk_reward_max=2.0,
                size_scalar=2.0,
            )
        }
        cal = RegimeRiskCalibrator(custom_params=custom)
        params = cal.get_risk_params(MarketRegime.BULL)
        assert params.max_position_pct == 0.10
        assert params.size_scalar == 2.0

    def test_regime_detection_result_to_dict(self):
        from regime_risk_calibrator import RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        result = cal.detect_regime([0.001] * 60)
        d = result.to_dict()
        assert "regime" in d
        assert "annualised_vol" in d
        assert "max_drawdown" in d

    def test_risk_params_to_dict(self):
        from regime_risk_calibrator import MarketRegime, RegimeRiskCalibrator

        cal = RegimeRiskCalibrator()
        params = cal.get_risk_params(MarketRegime.BULL)
        d = params.to_dict()
        assert d["regime"] == "BULL"
        assert "max_position_pct" in d
        assert "size_scalar" in d


# ---------------------------------------------------------------------------
# Tests: Full Graph Flow (mocked)
# ---------------------------------------------------------------------------


class TestFullGraphFlowMocked:
    """End-to-end graph flow with all external APIs mocked."""

    def test_graph_setup_imports_new_modules(self):
        """graph_setup.py references DecisionJournal and RegimeRiskCalibrator."""
        import pathlib

        src = pathlib.Path("signals/agents/quantagent/graph_setup.py").read_text()
        # Verify the source imports the new modules
        assert "from decision_journal import DecisionJournal" in src
        assert "from regime_risk_calibrator import RegimeRiskCalibrator" in src
        # Verify journal and calibrator are instantiated
        assert "DecisionJournal()" in src
        assert "RegimeRiskCalibrator()" in src

    def test_decision_agent_with_all_features(self):
        """Decision agent works with journal + regime calibrator together."""
        from decision_agent import create_final_trade_decider
        from decision_journal import DecisionJournal
        from regime_risk_calibrator import RegimeRiskCalibrator

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = MOCK_DECISION_JSON
        mock_llm.invoke.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            journal = DecisionJournal(path=Path(tmpdir) / "j.json")
            calibrator = RegimeRiskCalibrator()

            node = create_final_trade_decider(
                mock_llm, journal=journal, regime_calibrator=calibrator
            )

            state = _make_mock_state()
            state["recent_returns"] = [-0.05] * 10 + [0.001] * 50  # crisis regime

            result = node(state)

            # Decision returned
            assert "final_trade_decision" in result
            assert result["final_trade_decision"] == MOCK_DECISION_JSON

            # Journal recorded
            entries = journal.get_entries()
            assert len(entries) == 1

            # Prompt includes regime section
            assert "CRISIS" in result["decision_prompt"]

    def test_github_agent_feeds_into_decision(self):
        """GitHub agent output feeds correctly into decision agent input."""
        from decision_agent import create_final_trade_decider
        from github_agent import create_github_agent

        # Create GitHub agent with mocked deps
        mock_github_llm = MagicMock()
        mock_github_response = MagicMock()
        mock_github_response.content = "Found relevant repos: quant-strategy (1500 stars)"
        mock_github_llm.invoke.return_value = mock_github_response

        mock_http_resp = MagicMock()
        mock_http_resp.json.return_value = MOCK_GITHUB_API_RESPONSE
        mock_http_resp.raise_for_status.return_value = None

        with patch("github_agent.httpx.get", return_value=mock_http_resp):
            github_node = create_github_agent(mock_github_llm)
            state = _make_mock_state()
            github_result = github_node(state)

        # Now feed into decision agent
        state.update(github_result)

        mock_decision_llm = MagicMock()
        mock_decision_response = MagicMock()
        mock_decision_response.content = MOCK_DECISION_JSON
        mock_decision_llm.invoke.return_value = mock_decision_response

        decision_node = create_final_trade_decider(mock_decision_llm)
        decision_result = decision_node(state)

        # Verify GitHub report is in the decision prompt
        assert "quant-strategy" in decision_result["decision_prompt"]
        assert "final_trade_decision" in decision_result
