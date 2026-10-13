"""离线确定性 Mock 推理后端。

用途：在没有 GPU、没有网络的机器上验证整条工程链路（切片 → 检索 → 重排 →
编排 → 引用校验 → 评测指标），并保证 CI 结果可复现。

实现要点：
- 向量：字符 n-gram 哈希到固定维度，确定性、零依赖；
- 生成：从提示词中抽取检索上下文里的引用标记，模拟"有依据才作答、无依据则拒答"。
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.llm.base import ChatResult, Message, estimate_tokens

DIM = 256
CITATION_RE = re.compile(r"【([^】]{1,64})】")
# 提示词里的段落标题不是引用标记，mock 必须忽略，避免"把指令当证据"
RESERVED_SECTIONS = {
    "用户问题",
    "检索证据",
    "工具输出",
    "补充信息",
    "用户角色",
    "历史对话摘要",
    "检索证据】（引用标记必须原样复制）",
}
REFUSAL = "未在知识库中找到足以回答该问题的依据，我不作推测。请补充更具体的文件或条款范围。"


class MockLLM:
    """确定性 mock：同样的输入永远得到同样的输出。"""

    name = "mock"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ 向量
    @staticmethod
    def _grams(text: str) -> list[str]:
        lowered = text.lower()
        grams: list[str] = re.findall(r"[a-z0-9]+", lowered)
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
        grams.extend(cjk)
        grams.extend(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
        return grams or ["<empty>"]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * DIM
            for gram in self._grams(text):
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % DIM
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    # ------------------------------------------------------------------ 生成
    def chat(
        self,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        # 引用标记只从"非 system"内容中提取：system 是格式说明，不是证据来源
        content = "\n".join(
            str(message.get("content", "")) for message in messages if message.get("role") != "system"
        )
        question = self._last_user(messages)
        text = self._judge(prompt) if self._is_judge(prompt) else self._answer(content, question)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ChatResult(
            text=text,
            provider=self.name,
            model="mock-chat",
            latency_ms=latency_ms,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
        )

    @staticmethod
    def _last_user(messages: Sequence[Message]) -> str:
        for message in reversed(list(messages)):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _is_judge(prompt: str) -> bool:
        lowered = prompt.lower()
        return "评分" in prompt or ("score" in lowered and "json" in lowered)

    @staticmethod
    def _judge(prompt: str) -> str:
        # 评测环节的 LLM-as-Judge：mock 固定给中等偏上分，保证离线可跑
        return '{"score": 0.9, "reason": "mock judge：答案与检索依据一致，未见明显编造。"}'

    def _answer(self, prompt: str, question: str) -> str:
        citations: list[str] = []
        for match in CITATION_RE.finditer(prompt):
            citation = match.group(1).strip()
            if not citation or citation in citations:
                continue
            if any(citation.startswith(section) for section in RESERVED_SECTIONS):
                continue
            citations.append(citation)
        if not citations:
            return REFUSAL
        top = citations[:3]
        lines = [f"1. 依据【{top[0]}】的规定，该事项应按条文要求执行并留存记录。"]
        for offset, citation in enumerate(top[1:], start=2):
            lines.append(f"{offset}. 同时参见【{citation}】的对应条款作为补充依据。")
        lines.append("")
        lines.append("引用来源：" + "、".join(f"【{c}】" for c in top))
        lines.append(f"（离线 mock 生成，仅用于验证链路；真实作答由 {self.settings.resolved_chat_model} 产生）")
        return "\n".join(lines)

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "ok": True,
            "detail": "离线确定性后端（无网络、无 GPU 依赖）",
        }
