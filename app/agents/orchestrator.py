"""主智能体：意图路由 → 计划 → 派发子智能体 → 引用校验 → 兜底降级。

为什么不让一个大模型端到端处理：
1. 路由是确定性的（关键词命中优先，模型分类可选），审计场景要求结论可复现；
2. 每个子任务有独立证据边界，串行执行时证据可校验、可溯源；
3. 校验失败时降级为「拒答」，而不是输出无依据的结论——这是合规场景的底线。
"""
from __future__ import annotations

from typing import Any

from app.agents.prompts import REFUSAL_TEXT
from app.agents.state import AgentState
from app.agents.subagents import CalcAgent, DocAgent, RuleAgent, SubAgentResult, extract_citations
from app.config import Settings, get_settings
from app.llm.base import LLMClient
from app.llm.factory import build_llm
from app.obs.tracer import Tracer
from app.rag.retriever import Retriever
from app.rag.store import LocalHybridIndex
from app.tools.registry import ToolRegistry, build_default_registry

# 路由信号分两级：强信号是"动作明确"，弱信号是"领域词汇"。
# 关键教训：只看领域词汇会把"关于抽样的问题"错判为"要算抽样"，
# 从而跳过检索、答非所问——强信号优先，弱信号只在两者都命中时判为 mixed。
STRONG_CALC = (
    "计算",
    "算一下",
    "帮我算",
    "抽样方案",
    "抽样计划",
    "样本量是",
    "抽样间隔是",
    "是多少",
    "分别为多少",
    "重要性水平为",
)
STRONG_DOC = ("底稿", "台账", "目录", "制度", "附件", "合同", "文件", "查看", "读取", "抽取出")
RULE_HINTS = ("准则", "条款", "依据", "规定", "要求", "适用", "关系", "如何", "应当", "为什么", "是否")
PLAN_BY_INTENT: dict[str, tuple[str, ...]] = {
    "rule": ("rule_agent",),
    "doc": ("doc_agent",),
    "calc": ("calc_agent",),
    "mixed": ("rule_agent", "doc_agent", "calc_agent"),
}
REFUSAL_MARKERS = ("依据不足", "未在知识库中找到", "不作推测", "无法回答", "依据不足以")


class Orchestrator:
    """主智能体：负责任务分解、派发与质量兜底。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        retriever: Retriever | None = None,
        registry: ToolRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.registry = registry or build_default_registry(str(self.settings.workspace_dir))
        self.retriever = retriever or Retriever(LocalHybridIndex(), self.llm, self.settings)

    # ------------------------------------------------------------------ 路由
    def route(self, question: str) -> str:
        """确定性意图路由：强信号决定单域路由，跨域命中则为 mixed 串行编排。"""
        calc = any(keyword in question for keyword in STRONG_CALC)
        doc = any(keyword in question for keyword in STRONG_DOC)
        rule = any(keyword in question for keyword in RULE_HINTS)
        if calc and doc:
            return "mixed"
        if calc:
            return "calc"
        if doc and rule:
            return "mixed"
        if doc:
            return "doc"
        return "rule"

    def _agents(self) -> dict[str, Any]:
        return {
            "rule_agent": RuleAgent(self.retriever, self.llm, self.settings),
            "doc_agent": DocAgent(self.llm, self.registry, self.settings),
            "calc_agent": CalcAgent(self.llm, self.registry, self.settings),
        }

    # ------------------------------------------------------------------ 执行
    def run(
        self,
        question: str,
        role: str = "assistant",
        top_k: int | None = None,
        tracer: Tracer | None = None,
    ) -> AgentState:
        tracer = tracer or Tracer(self.settings, name="orchestrator.run")
        state = AgentState(question=question, role=role)
        with tracer.span("route"):
            state.intent = self.route(question)
        tracer.event("intent", intent=state.intent, role=role)

        agents = self._agents()
        plan = PLAN_BY_INTENT.get(state.intent, ("rule_agent",))
        for name in plan:
            state.planned(name, name)

        answers: list[str] = []
        for name in plan:
            with tracer.span(name) as span:
                result: SubAgentResult = agents[name].handle(state, top_k=top_k)
                span.attributes.update({"answer_chars": len(result.answer), "citations": len(result.citations)})
            for tool_result in result.tool_results:
                state.add_tool_result(tool_result)
            usage = result.detail.get("usage") or {}
            state.add_usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            state.add_citations(result.citations)
            retrieval = result.detail.get("retrieval")
            if retrieval is not None and state.retrieval is None:
                state.retrieval = retrieval
            if result.answer:
                answers.append(result.answer)
            state.complete(name, f"引用 {len(result.citations)} 条")

        state.answer = self._merge(answers)
        with tracer.span("verify", allowed=len(state.evidence_citations)):
            self.verify(state)
        return state

    def answer(
        self, question: str, role: str = "assistant", top_k: int | None = None
    ) -> tuple[AgentState, dict[str, Any]]:
        """一次性问答：返回状态与 trace 记录（供 API、脚本与评测复用）。"""
        tracer = Tracer(self.settings, name="orchestrator.run")
        state = self.run(question, role=role, top_k=top_k, tracer=tracer)
        record = tracer.finish(
            output={
                "intent": state.intent,
                "verified": state.verified,
                "refused": state.refused,
                "citations": state.citations,
                "usage": state.usage,
            }
        )
        return state, record

    @staticmethod
    def _merge(answers: list[str]) -> str:
        cleaned = [answer.strip() for answer in answers if answer and answer.strip()]
        if not cleaned:
            return REFUSAL_TEXT
        return "\n\n".join(cleaned)

    # ------------------------------------------------------------------ 校验
    def verify(self, state: AgentState) -> None:
        """引用校验：回答中出现的每个引用标记都必须能溯源，否则判定不合规。"""
        allowed = set(state.evidence_citations)
        structured = [citation for citation in extract_citations(state.answer) if "·" in citation]
        unknown = [citation for citation in structured if citation not in allowed]
        if unknown:
            state.verification_notes.append(f"存在无法溯源的引用标记：{'、'.join(unknown)}")

        if not structured:
            if any(marker in state.answer for marker in REFUSAL_MARKERS):
                state.refused = True
                state.verification_notes.append("证据不足，按规则拒答（不输出无依据结论）")
            else:
                state.verification_notes.append("回答未包含任何引用标记，判定为不合规输出")

        state.verified = not unknown and (bool(structured) or state.refused)
        if not state.verified:
            state.answer = REFUSAL_TEXT + "\n\n（质量校验未通过：" + "；".join(state.verification_notes) + "）"

