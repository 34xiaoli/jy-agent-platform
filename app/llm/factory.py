"""推理后端工厂：按 LLM_PROVIDER 组装客户端。

切换成本为零：同一份业务代码，改一个环境变量即可在
Ollama（离线单机）↔ vLLM（生产并发）↔ 云 API ↔ mock（离线测试）之间切换。
"""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import LLMClient, LLMError
from app.llm.mock_client import MockLLM
from app.llm.ollama_client import OllamaClient
from app.llm.openai_compat import OpenAICompatClient


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.provider
    if provider == "ollama":
        return OllamaClient(settings)
    if provider == "vllm":
        return OpenAICompatClient(settings, provider_name="vllm")
    if provider == "openai":
        return OpenAICompatClient(settings, provider_name="openai")
    if provider == "mock":
        return MockLLM(settings)
    raise LLMError(
        f"未知的 LLM_PROVIDER：{provider}（可选：ollama / vllm / openai / mock）"
    )


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """进程内单例：避免每次请求重建连接与客户端。"""
    return build_llm()


def reset_llm() -> None:
    get_llm.cache_clear()
