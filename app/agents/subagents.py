"""子智能体：按「任务域」拆分，而不是按部门或系统拆分。

三个子智能体：
- RuleAgent：回答「依据是什么」——只依赖 L1–L3 检索，必须给出条款级引用；
- DocAgent：回答「事实是什么」——只读文件，抽取金额、科目、日期等事实字段；
- CalcAgent：回答「数字是多少」——参数解析 + 调用确定性工具，模型只解释结果、不负责算数。

拆分理由（也是面试最常被追问的点）：
1. 任务域差异大：检索式回答 / 事实抽取 / 确定性计算，所需的上下文与校验方式完全不同；
2. 上下文污染：把长底稿塞进准则问答会稀释注意力，反过来同样；
3. 独立评测：每个子智能体可单独建评测集，否则整体指标无法归因。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.agents.prompts import REFUSAL_TEXT, SYSTEM_PROMPT, build_answer_prompt
from app.agents.state import AgentState
from app.config import Settings, get_settings
from app.knowledge.layers import allowed_layers
from app.llm.base import LLMClient
from app.rag.retriever import Retriever
from app.tools.registry import ToolRegistry, ToolResult

CITATION_RE = re.compile(r"【([^】]{1,64})】")
AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|亿|万元|万|元)?")
UNIT_FACTOR = {"亿元": 1e8, "亿": 1e8, "万元": 1e4, "万": 1e4, "元": 1.0, "": 1.0}
FILE_LINE_RE = re.compile(r"^-\s+(.+?)（", re.M)
NUMBER_LIST_RE = re.compile(r"\[[\d,\s\.]+\]|(?:\d+(?:\.\d+)?\s*[,，、]\s*)+\d+(?:\.\d+)?")


def extract_citations(text: str) -> list[str]:
    """抽取回答中的引用标记（去重、保序）。"""
    seen: list[str] = []
    for match in CITATION_RE.finditer(text or ""):
        citation = match.group(1).strip()
        if citation and citation not in seen:
            seen.append(citation)
    return seen


def extract_amounts(text: str) -> list[float]:
    """从问题中抽取金额（支持万元 / 亿元单位）。"""
    amounts: list[float] = []
    for value, unit in AMOUNT_RE.findall(text or ""):
        try:
            number = float(value)
        except ValueError:
            continue
        if number <= 0:
            continue
        amounts.append(number * UNIT_FACTOR.get(unit or "", 1.0))
    return amounts


def extract_number_list(text: str) -> list[float]:
    """抽取形如 [120, 80, 45] 或 120, 80, 45 的金额序列。"""
    match = NUMBER_LIST_RE.search(text or "")
    if not match:
        return []
    return [float(item) for item in re.findall(r"\d+(?:\.\d+)?", match.group(0))]


@dataclass
class SubAgentResult:
    name: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


class RuleAgent:
    """准则检索子智能体：证据即条款，引用必须可溯源。"""

    name = "rule_agent"
    description = "准则 / 法规条款检索问答"

    def __init__(self, retriever: Retriever, llm: LLMClient, settings: Settings | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()

    def handle(self, state: AgentState, top_k: int | None = None) -> SubAgentResult:
        layers = allowed_layers(state.role)
        retrieval = self.retriever.retrieve(state.question, top_k=top_k, layers=layers)
        if not retrieval.items:
            return SubAgentResult(
                name=self.name,
                answer=REFUSAL_TEXT,
                detail={"retrieval": retrieval, "reason": "empty_retrieval"},
            )
        top_coverage = float(retrieval.items[0].detail.get("coverage", 0.0))
        if self.settings.min_top_coverage > 0 and top_coverage < self.settings.min_top_coverage:
            return SubAgentResult(
                name=self.name,
                answer=REFUSAL_TEXT,
                detail={
                    "retrieval": retrieval,
                    "reason": f"low_coverage={top_coverage:.3f} < {self.settings.min_top_coverage}",
                },
            )
        prompt = build_answer_prompt(state, evidence=retrieval.context, tool_output="")
        result = self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=self.settings.max_tokens,
        )
        allowed = set(retrieval.citations)
        citations = [item for item in extract_citations(result.text) if item in allowed]
        return SubAgentResult(
            name=self.name,
            answer=result.text,
            citations=citations,
            detail={
                "retrieval": retrieval,
                "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
                "layers": layers,
            },
        )


class DocAgent:
    """文档事实抽取子智能体：先确定性定位文件，再让模型抽取字段。"""

    name = "doc_agent"
    description = "底稿 / 制度文件事实抽取"

    def __init__(self, llm: LLMClient, registry: ToolRegistry, settings: Settings | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings()

    def _pick_file(self, question: str, listing: str) -> str:
        files = FILE_LINE_RE.findall(listing)
        if not files:
            return ""
        keywords = re.findall(r"[\u4e00-\u9fff]{2,}", question)
        best, best_score = files[0], -1
        for path in files:
            score = sum(1 for keyword in keywords if keyword in path)
            if score > best_score:
                best, best_score = path, score
        return best

    def handle(self, state: AgentState, top_k: int | None = None) -> SubAgentResult:
        listing = self.registry.call("list_working_files", {})
        tool_results = [listing]
        if not listing.ok:
            return SubAgentResult(name=self.name, answer=f"文件列表获取失败：{listing.error}", tool_results=tool_results)

        target = self._pick_file(state.question, listing.output)
        if not target:
            return SubAgentResult(
                name=self.name,
                answer="工作目录中没有可用文件，无法抽取事实。",
                tool_results=tool_results,
            )

        reading = self.registry.call("read_text_file", {"path": target})
        tool_results.append(reading)
        tool_output = f"【TOOL·read_text_file】（文件：{target}）\n{reading.output}"
        prompt = build_answer_prompt(state, evidence="", tool_output=tool_output)
        result = self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=self.settings.max_tokens,
        )
        return SubAgentResult(
            name=self.name,
            answer=result.text,
            citations=["TOOL·read_text_file"],
            tool_results=tool_results,
            detail={
                "file": target,
                "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
            },
        )


class CalcAgent:
    """计算子智能体：确定性工具负责算，模型只负责解释。"""

    name = "calc_agent"
    description = "重要性水平 / 抽样等确定性计算"

    def __init__(self, llm: LLMClient, registry: ToolRegistry, settings: Settings | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings()

    def decide_tool(self, question: str) -> tuple[str, dict[str, Any]]:
        """参数解析（规则版）：生产环境可替换为 function calling，本仓库保留确定性回退。"""
        amounts = extract_number_list(question) or extract_amounts(question)
        if "抽样" in question or "样本" in question:
            sample_size = 0
            for number in extract_amounts(question):
                if number == int(number) and 1 <= number <= 200:
                    sample_size = int(number)
            sample_size = sample_size or max(1, min(len(amounts) or 10, 10))
            return "monetary_unit_sampling", {"amounts": amounts or [100.0, 50.0, 20.0], "sample_size": sample_size}
        if "分层" in question:
            return "stratified_sample", {"amounts": amounts or [100.0, 50.0, 20.0], "sample_size": 3, "strata": 3}
        benchmark = max(amounts) if amounts else 100000000.0
        rate = 0.005 if ("收入" in question or "营业收入" in question) else 0.05
        return "calc_materiality", {"benchmark": benchmark, "rate": rate}

    def handle(self, state: AgentState, top_k: int | None = None) -> SubAgentResult:
        tool_name, arguments = self.decide_tool(state.question)
        result = self.registry.call(tool_name, arguments)
        tool_results = [result]
        if not result.ok:
            return SubAgentResult(name=self.name, answer=f"计算未完成：{result.error}", tool_results=tool_results)
        tool_output = f"【TOOL·{tool_name}】参数 {json.dumps(arguments, ensure_ascii=False)}\n{result.output}"
        prompt = build_answer_prompt(
            state,
            evidence="",
            tool_output=tool_output,
            extra="所有数字必须原样引用工具输出，不要自行重算或四舍五入。",
        )
        chat = self.llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.settings.max_tokens,
        )
        return SubAgentResult(
            name=self.name,
            answer=chat.text,
            citations=[f"TOOL·{tool_name}"],
            tool_results=tool_results,
            detail={
                "tool": tool_name,
                "arguments": arguments,
                "usage": {"prompt_tokens": chat.prompt_tokens, "completion_tokens": chat.completion_tokens},
            },
        )

