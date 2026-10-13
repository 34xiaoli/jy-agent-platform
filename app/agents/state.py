"""智能体状态：显式状态对象，便于观测、复现与单步调试。

实现说明：此处用「显式状态机」实现编排（节点 = 纯函数，状态 = dataclass）。
之所以不依赖 LangGraph：本仓库要求零三方依赖也能跑通全链路；两者的状态模型等价，
迁移时只需把节点函数接到 StateGraph 的 add_node 上即可（见 docs/architecture.md）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.rag.retriever import RetrievalResult
from app.tools.registry import ToolResult


@dataclass
class PlanStep:
    name: str
    agent: str
    status: str = "pending"
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "agent": self.agent, "status": self.status, "detail": self.detail}


@dataclass
class AgentState:
    """一次问答的完整状态：可序列化、可回放、可被评测脚本消费。"""

    question: str
    role: str = "assistant"
    intent: str = "unknown"
    steps: list[PlanStep] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    verified: bool = False
    refused: bool = False
    verification_notes: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    # ------------------------------------------------------------------ 工具
    def planned(self, name: str, agent: str) -> None:
        self.steps.append(PlanStep(name=name, agent=agent))

    def complete(self, name: str, detail: str = "") -> None:
        for step in self.steps:
            if step.name == name and step.status == "pending":
                step.status = "done"
                step.detail = detail
                return

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage["prompt_tokens"] += max(0, prompt_tokens)
        self.usage["completion_tokens"] += max(0, completion_tokens)

    def add_citations(self, citations: list[str]) -> None:
        for citation in citations:
            if citation and citation not in self.citations:
                self.citations.append(citation)

    @property
    def evidence_citations(self) -> list[str]:
        """本次允许出现的引用标记集合（检索证据 + 工具输出）。"""
        allowed: list[str] = []
        if self.retrieval:
            allowed.extend(self.retrieval.citations)
        allowed.extend(f"TOOL·{result.name}" for result in self.tool_results if result.ok)
        return allowed

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "role": self.role,
            "intent": self.intent,
            "steps": [step.as_dict() for step in self.steps],
            "retrieval": self.retrieval.as_dict() if self.retrieval else None,
            "tool_results": [result.as_dict() for result in self.tool_results],
            "answer": self.answer,
            "citations": self.citations,
            "verified": self.verified,
            "refused": self.refused,
            "verification_notes": self.verification_notes,
            "usage": self.usage,
        }
