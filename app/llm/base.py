"""LLM 抽象层：RAG、工具、智能体只依赖这里的接口，不依赖任何厂商 SDK。

工程价值：
1. 推理后端可切换（Ollama 本地离线 / vLLM 生产并发 / 云 API），业务代码零改动；
2. 评测与单元测试可注入 mock，无需 GPU 与网络，CI 可复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

Message = dict[str, str]


class LLMError(RuntimeError):
    """统一的推理调用异常，便于上层做重试与降级。"""


@dataclass
class ChatResult:
    """一次生成的完整结果（含可观测性字段）。"""

    text: str
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "answer_chars": len(self.text),
        }


@runtime_checkable
class LLMClient(Protocol):
    """推理后端协议。实现方需保证 chat / embed / health 三个方法。"""

    name: str

    def chat(
        self,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:  # pragma: no cover - 协议声明
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        ...

    def health(self) -> dict[str, Any]:  # pragma: no cover
        ...


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1 字 1 token，ASCII 约 4 字符 1 token。

    仅在缺少 usage 返回（如部分自建推理服务）时用于成本观测。
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    cjk_chars = len(text) - ascii_chars
    return int(cjk_chars + ascii_chars / 4) + 1
