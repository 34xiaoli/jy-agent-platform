"""上下文工程：记忆分层 + 压缩 + 预算裁剪。

三类信息必须分开管（这正是「上下文工程」与「提示词工程」的分界）：
1. 短期记忆：最近 N 轮对话原文，保证多轮指代可解；
2. 长期摘要：超过阈值后由模型压缩成要点，防止历史无限膨胀；
3. 任务上下文：检索证据 + 工具结果，按预算裁剪与去重，证据优先级高于历史。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.base import LLMClient, Message

SUMMARY_PROMPT = (
    "请把以下多轮问答压缩为不超过 150 字的要点，必须保留：业务对象、金额或科目、"
    "已确认的结论、待办事项。只输出要点本身，不要解释。\n\n{history}"
)


@dataclass
class Turn:
    role: str
    content: str


class ConversationMemory:
    """会话记忆：滑动窗口 + 超阈值触发摘要压缩。"""

    def __init__(
        self,
        llm: LLMClient,
        max_turns: int = 6,
        summarize_at: int = 8,
        summary_chars: int = 400,
    ) -> None:
        self.llm = llm
        self.max_turns = max_turns
        self.summarize_at = summarize_at
        self.summary_chars = summary_chars
        self._turns: list[Turn] = []
        self._summary: str = ""

    def add(self, role: str, content: str) -> None:
        if not content:
            return
        self._turns.append(Turn(role=role, content=content))
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        if len(self._turns) <= self.summarize_at:
            return
        overflow = self._turns[: -self.max_turns]
        self._turns = self._turns[-self.max_turns :]
        history = "\n".join(f"{turn.role}: {turn.content[:300]}" for turn in overflow)
        try:
            result = self.llm.chat(
                [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}],
                temperature=0.0,
                max_tokens=256,
            )
            summary = result.text.strip()
        except Exception:  # noqa: BLE001 - 摘要失败时退化为截断，保证主链路可用
            summary = history[: self.summary_chars]
        self._summary = f"{self._summary}\n{summary}".strip()[: self.summary_chars * 2]

    @property
    def summary(self) -> str:
        return self._summary

    def history(self) -> list[Message]:
        messages: list[Message] = []
        if self._summary:
            messages.append({"role": "system", "content": f"【历史对话摘要】\n{self._summary}"})
        messages.extend({"role": turn.role, "content": turn.content} for turn in self._turns)
        return messages

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""


@dataclass
class AssemblyResult:
    text: str
    used_chars: int = 0
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


@dataclass
class ContextBudget:
    """按字符预算组装上下文：优先级高的先保留，超预算的显式丢弃并留痕。"""

    max_chars: int = 6000

    def assemble(self, sections: list[tuple[str, int, str]]) -> AssemblyResult:
        """sections: [(名称, 优先级, 文本)]，优先级越大越先保留。"""
        result = AssemblyResult(text="")
        blocks: list[str] = []
        for name, _priority, text in sorted(sections, key=lambda item: item[1], reverse=True):
            if not text:
                continue
            if result.used_chars + len(text) > self.max_chars and blocks:
                result.dropped.append(name)
                continue
            blocks.append(f"### {name}\n{text}")
            result.kept.append(name)
            result.used_chars += len(text)
        result.text = "\n\n".join(blocks)
        return result
