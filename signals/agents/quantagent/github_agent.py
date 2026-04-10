"""
Agent for GitHub repository research in the context of quantitative trading.
Uses LLM to discover and evaluate relevant open-source repositories
(trading strategies, risk models, data pipelines) related to the asset under analysis.
"""

import json
import os
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# GitHub search helper
# ---------------------------------------------------------------------------

_GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
_GITHUB_HEADERS = {"Accept": "application/vnd.github+json"}


def _github_token() -> str | None:
    """Return a GitHub PAT from the environment, if available."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _search_github_repos(
    query: str,
    max_results: int = 5,
    sort: str = "stars",
    order: str = "desc",
) -> list[dict[str, Any]]:
    """
    Search GitHub for repositories matching *query*.

    Returns a simplified list of dicts with keys:
        name, full_name, description, stars, language, url, updated_at
    """
    headers = dict(_GITHUB_HEADERS)
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {"q": query, "sort": sort, "order": order, "per_page": max_results}

    try:
        resp = httpx.get(_GITHUB_SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[github_agent] GitHub API error: {exc}")
        return []

    repos: list[dict[str, Any]] = []
    for item in data.get("items", [])[:max_results]:
        repos.append(
            {
                "name": item.get("full_name", ""),
                "full_name": item.get("full_name", ""),
                "description": (item.get("description") or "")[:200],
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language", ""),
                "url": item.get("html_url", ""),
                "updated_at": item.get("updated_at", ""),
            }
        )
    return repos


# ---------------------------------------------------------------------------
# Build search queries from the trading context
# ---------------------------------------------------------------------------


def _build_search_queries(stock_name: str) -> list[str]:
    """
    Return a list of GitHub search queries derived from the stock / asset name
    and common quantitative-finance topics.
    """
    base_topics = [
        "quantitative trading strategy",
        "algorithmic trading",
        "financial analysis",
    ]
    queries = []
    for topic in base_topics:
        queries.append(f"{stock_name} {topic}")
    # Always include a generic quant-finance query
    queries.append("quantitative finance python")
    return queries


# ---------------------------------------------------------------------------
# Agent node factory
# ---------------------------------------------------------------------------


def create_github_agent(llm):
    """
    Create a GitHub research agent node for the LangGraph trading graph.

    The node:
    1. Searches GitHub for repositories relevant to the asset under analysis.
    2. Asks the LLM to synthesise a short research report from the discovered repos.
    3. Returns ``github_report`` (str) and ``github_repos`` (list[dict]) in the state.

    Args:
        llm: A LangChain chat model used to generate the summary report.

    Returns:
        A callable ``github_agent_node(state) -> dict`` suitable for
        ``StateGraph.add_node``.
    """

    def github_agent_node(state: dict) -> dict:
        stock_name = state.get("stock_name", "UNKNOWN")
        indicator_report = state.get("indicator_report", "")
        pattern_report = state.get("pattern_report", "")
        trend_report = state.get("trend_report", "")

        # --- Step 1: Search GitHub ---
        all_repos: list[dict[str, Any]] = []
        seen_names: set = set()
        queries = _build_search_queries(stock_name)

        for q in queries:
            results = _search_github_repos(q, max_results=3)
            for repo in results:
                if repo["name"] not in seen_names:
                    seen_names.add(repo["name"])
                    all_repos.append(repo)

        # Keep at most 10 repos overall
        all_repos = all_repos[:10]

        # --- Step 2: Ask LLM to summarise ---
        repos_text = json.dumps(all_repos, indent=2) if all_repos else "No repositories found."

        system_prompt = (
            "You are a quantitative-finance research assistant. "
            "You are given a list of GitHub repositories discovered via search, "
            "along with the current technical analysis context for a stock/asset. "
            "Your task is to write a concise research report (3-5 paragraphs) that:\n"
            "1. Highlights the most relevant repositories for the asset being analysed.\n"
            "2. Explains how each relevant repo could support or inform the trading decision.\n"
            "3. Notes any open-source risk models, backtesting frameworks, or data pipelines "
            "that could be useful.\n"
            "Be specific and actionable. Cite repository names and star counts."
        )

        human_prompt = (
            f"Asset under analysis: {stock_name}\n\n"
            f"--- Technical Indicator Report ---\n{indicator_report}\n\n"
            f"--- Pattern Report ---\n{pattern_report}\n\n"
            f"--- Trend Report ---\n{trend_report}\n\n"
            f"--- Discovered GitHub Repositories ---\n{repos_text}\n\n"
            "Please write the GitHub research report."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = llm.invoke(messages)
            github_report = response.content
        except Exception as exc:
            print(f"[github_agent] LLM error: {exc}")
            github_report = (
                f"GitHub research could not be completed due to an error: {exc}"
            )

        return {
            "github_report": github_report,
            "github_repos": all_repos,
            "messages": state.get("messages", []) + messages,
        }

    return github_agent_node
