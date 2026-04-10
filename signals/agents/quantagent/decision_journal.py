"""
Decision Journal — logs every trade decision with input data, score, and state.

Each decision is recorded as a structured entry containing:
- Timestamp
- Stock name and time frame
- Input reports (indicator, pattern, trend, GitHub)
- Decision output (direction, confidence, risk-reward ratio, justification)
- Graph state snapshot

Usage:
    journal = DecisionJournal()
    journal.record(state, decision_output)
    entries = journal.get_entries()
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Journal entry
# ---------------------------------------------------------------------------


@dataclass
class DecisionEntry:
    """Single decision log entry."""

    timestamp: str
    stock_name: str
    time_frame: str

    # Input reports (truncated for storage)
    indicator_report: str
    pattern_report: str
    trend_report: str
    github_report: str

    # Decision output
    decision_raw: str
    decision_direction: str
    decision_confidence: float
    decision_risk_reward: float
    decision_justification: str

    # State snapshot
    state_keys: list[str] = field(default_factory=list)
    github_repos_count: int = 0
    messages_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Decision Journal
# ---------------------------------------------------------------------------

_MAX_REPORT_CHARS = 2000  # truncate long reports for storage


class DecisionJournal:
    """
    Persistent journal that records every trade decision.

    Entries are stored in-memory and optionally persisted to a JSON file.
    """

    DEFAULT_PATH = Path("logs/decision_journal.json")

    def __init__(self, path: Path | None = None, max_entries: int = 500) -> None:
        self.path = path or self.DEFAULT_PATH
        self.max_entries = max_entries
        self._entries: list[DecisionEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing entries from disk."""
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            for item in data:
                self._entries.append(DecisionEntry(**item))
        except Exception:
            # Start fresh on any parse error
            self._entries = []

    def _save(self) -> None:
        """Persist entries to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = [e.to_dict() for e in self._entries]
            self.path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass  # best-effort persistence

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, state: dict[str, Any], decision_output: str) -> DecisionEntry:
        """
        Record a decision from the graph state and the raw LLM output.

        Args:
            state: The full LangGraph state dict after decision.
            decision_output: Raw text output from the decision LLM.

        Returns:
            The created DecisionEntry.
        """
        now = datetime.now(UTC).isoformat()

        # Parse decision fields from raw output
        direction, confidence, risk_reward, justification = self._parse_decision(
            decision_output
        )

        entry = DecisionEntry(
            timestamp=now,
            stock_name=str(state.get("stock_name", "UNKNOWN")),
            time_frame=str(state.get("time_frame", "UNKNOWN")),
            indicator_report=self._truncate(
                str(state.get("indicator_report", ""))
            ),
            pattern_report=self._truncate(
                str(state.get("pattern_report", ""))
            ),
            trend_report=self._truncate(str(state.get("trend_report", ""))),
            github_report=self._truncate(
                str(state.get("github_report", ""))
            ),
            decision_raw=self._truncate(decision_output),
            decision_direction=direction,
            decision_confidence=confidence,
            decision_risk_reward=risk_reward,
            decision_justification=justification,
            state_keys=list(state.keys()),
            github_repos_count=len(state.get("github_repos", [])),
            messages_count=len(state.get("messages", [])),
        )

        self._entries.append(entry)

        # Evict oldest entries if over limit
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        self._save()
        return entry

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_entries(
        self,
        stock_name: str | None = None,
        last_n: int | None = None,
    ) -> list[DecisionEntry]:
        """
        Return journal entries, optionally filtered.

        Args:
            stock_name: Filter by stock name (case-insensitive).
            last_n: Return only the last N entries.

        Returns:
            List of DecisionEntry objects.
        """
        entries = list(self._entries)
        if stock_name:
            entries = [
                e
                for e in entries
                if e.stock_name.upper() == stock_name.upper()
            ]
        if last_n is not None:
            entries = entries[-last_n:]
        return entries

    def summary(self) -> dict[str, Any]:
        """Return an aggregate summary of all journal entries."""
        if not self._entries:
            return {
                "total_decisions": 0,
                "directions": {},
                "avg_confidence": 0.0,
                "stocks_traded": [],
            }

        directions: dict[str, int] = {}
        confidences: list[float] = []
        stocks: set[str] = set()

        for e in self._entries:
            directions[e.decision_direction] = (
                directions.get(e.decision_direction, 0) + 1
            )
            confidences.append(e.decision_confidence)
            stocks.add(e.stock_name)

        return {
            "total_decisions": len(self._entries),
            "directions": directions,
            "avg_confidence": sum(confidences) / len(confidences)
            if confidences
            else 0.0,
            "stocks_traded": sorted(stocks),
        }

    def clear(self) -> None:
        """Clear all entries (in-memory and on disk)."""
        self._entries = []
        self._save()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_chars: int = _MAX_REPORT_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "…[truncated]"

    @staticmethod
    def _parse_decision(
        raw: str,
    ) -> tuple[str, float, float, str]:
        """
        Best-effort extraction of direction, confidence,
        risk_reward_ratio, and justification from LLM output.
        """
        direction = "UNKNOWN"
        confidence = 0.0
        risk_reward = 0.0
        justification = ""

        try:
            text = raw.strip()
            # Strip markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
            data = json.loads(text)
            direction = str(data.get("decision", data.get("direction", "UNKNOWN")))
            confidence = float(data.get("confidence", 0.0))
            risk_reward = float(data.get("risk_reward_ratio", 0.0))
            justification = str(data.get("justification", ""))
        except (json.JSONDecodeError, ValueError, KeyError):
            # If raw output isn't JSON, store as-is
            justification = raw[:500] if raw else ""

        return direction, confidence, risk_reward, justification
