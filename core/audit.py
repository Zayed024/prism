"""Prism Audit Trail — tracks the full pipeline execution with cost estimation.

Unlike traditional audit logs that just record actions, Prism's audit captures
the multi-agent decision process: which tools each agent called, how they
negotiated, what the orchestrator kept/discarded, and the estimated cost.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime


# Approximate token costs for Gemini models on Vertex AI (per 1M tokens)
MODEL_COSTS = {
    "gemini-3-flash-preview": {"input": 0.30, "output": 1.20},
    "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


@dataclass
class AuditEntry:
    phase: str          # context, agent, negotiation, merge
    agent: str          # red, blue, green, orchestrator
    action: str         # tool_call, llm_generate, merge, etc.
    detail: str         # human-readable description
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: str = "success"  # success, error, skipped
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "agent": self.agent,
            "action": self.action,
            "detail": self.detail[:200],
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": self.latency_ms,
            "status": self.status,
            "timestamp": self.timestamp,
        }


class PrismAudit:
    """Collects audit entries across the full Prism pipeline."""

    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.entries: list[AuditEntry] = []
        self.model = model
        self.start_time = time.time()

    def log(self, phase: str, agent: str, action: str, detail: str,
            input_text: str = "", output_text: str = "",
            latency_ms: int = 0, status: str = "success"):
        """Log an audit entry with automatic cost estimation."""
        input_tokens = _estimate_tokens(input_text)
        output_tokens = _estimate_tokens(output_text)
        costs = MODEL_COSTS.get(self.model, MODEL_COSTS["gemini-3-flash-preview"])
        cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

        self.entries.append(AuditEntry(
            phase=phase, agent=agent, action=action, detail=detail,
            model=self.model, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, latency_ms=latency_ms, status=status,
            timestamp=datetime.now().isoformat(),
        ))

    def summary(self) -> dict:
        """Generate a cost and performance summary."""
        total_cost = sum(e.cost_usd for e in self.entries)
        total_input = sum(e.input_tokens for e in self.entries)
        total_output = sum(e.output_tokens for e in self.entries)
        total_time = int((time.time() - self.start_time) * 1000)

        by_phase = {}
        for e in self.entries:
            if e.phase not in by_phase:
                by_phase[e.phase] = {"cost": 0, "entries": 0, "latency_ms": 0}
            by_phase[e.phase]["cost"] += e.cost_usd
            by_phase[e.phase]["entries"] += 1
            by_phase[e.phase]["latency_ms"] += e.latency_ms

        by_agent = {}
        for e in self.entries:
            if e.agent not in by_agent:
                by_agent[e.agent] = {"cost": 0, "tool_calls": 0, "llm_calls": 0}
            by_agent[e.agent]["cost"] += e.cost_usd
            if e.action == "tool_call":
                by_agent[e.agent]["tool_calls"] += 1
            elif e.action == "llm_generate":
                by_agent[e.agent]["llm_calls"] += 1

        errors = [e for e in self.entries if e.status == "error"]

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_time_ms": total_time,
            "total_entries": len(self.entries),
            "errors": len(errors),
            "by_phase": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_phase.items()},
            "by_agent": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_agent.items()},
            "model": self.model,
        }

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]
