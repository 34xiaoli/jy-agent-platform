"""集中配置：全部配置项均可通过环境变量覆盖。

默认值的设计目标是「零三方依赖 + 离线 mock」也能完整跑通
切片 → 建索引 → 混合检索 → 重排 → 编排 → 引用校验 → 评测 全链路，
这样在任何机器（无 GPU、无网络）上都能验证工程正确性。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INDEX_DIR = DATA_DIR / "index"
TRACE_DIR = DATA_DIR / "traces"

# 各推理后端的默认模型（2026 年现役）。可用环境变量覆盖。
DEFAULT_CHAT_MODEL = {
    "ollama": "qwen3.5:9b",
    "vllm": "Qwen/Qwen3.5-9B-Instruct",
    "openai": "gpt-5.1-mini",
    "mock": "mock-chat",
}
DEFAULT_EMBED_MODEL = {
    "ollama": "bge-m3",
    "vllm": "Qwen/Qwen3-Embedding-0.6B",
    "openai": "text-embedding-3-large",
    "mock": "hash-ngram-256",
}
DEFAULT_BASE_URL = {
    "ollama": "http://127.0.0.1:11434",
    "vllm": "http://127.0.0.1:8000/v1",
    "openai": "https://api.openai.com/v1",
    "mock": "",
}


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _layer_prior(raw: str) -> dict[str, float]:
    prior: dict[str, float] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        try:
            prior[key.strip().upper()] = float(value)
        except ValueError:
            continue
    return prior or {"L1": 1.0, "L2": 0.96, "L3": 0.90, "L4": 0.84}


@dataclass(frozen=True)
class Settings:
    """运行期配置（不可变，便于并发安全地共享）。"""

    # --- 推理后端 ---
    provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "mock").lower())
    chat_model: str = field(default_factory=lambda: _str("CHAT_MODEL"))
    embed_model: str = field(default_factory=lambda: _str("EMBED_MODEL"))
    base_url: str = field(default_factory=lambda: _str("LLM_BASE_URL"))
    api_key: str = field(default_factory=lambda: _str("LLM_API_KEY"))
    request_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 60.0))
    temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1024))

    # --- 检索 ---
    recall_k: int = field(default_factory=lambda: _int("RECALL_K", 20))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    use_hybrid: bool = field(default_factory=lambda: _bool("USE_HYBRID", True))
    use_rerank: bool = field(default_factory=lambda: _bool("USE_RERANK", True))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    max_context_chars: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 6000))
    # 召回置信度闸门：首位候选的「词项覆盖率」低于该值时直接拒答。
    # 默认开启（0.30）：金标集实测分离区间为 (0.2121, 0.5484]，
    # 即"知识库无依据"类问题的最高覆盖率也只有 0.21，而"有依据"类最低为 0.55，
    # 0.30 落在区间中部，两侧都有足够余量。设 0 可关闭该闸门。
    min_top_coverage: float = field(default_factory=lambda: _float("MIN_TOP_COVERAGE", 0.30))
    layer_prior: dict[str, float] = field(
        default_factory=lambda: _layer_prior(_str("LAYER_PRIOR", "L1=1.00,L2=0.96,L3=0.90,L4=0.84"))
    )

    # --- 对外服务 ---
    service_api_key: str = field(default_factory=lambda: _str("SERVICE_API_KEY"))
    rate_limit_per_min: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MIN", 60))

    # --- 可观测性（Langfuse 为可选项；默认始终落 JSONL trace） ---
    enable_langfuse: bool = field(default_factory=lambda: _bool("ENABLE_LANGFUSE", False))
    langfuse_host: str = field(default_factory=lambda: _str("LANGFUSE_HOST", "http://127.0.0.1:3000"))
    langfuse_public_key: str = field(default_factory=lambda: _str("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = field(default_factory=lambda: _str("LANGFUSE_SECRET_KEY"))

    # --- 路径 ---
    knowledge_dir: Path = field(default_factory=lambda: Path(_str("KNOWLEDGE_DIR", str(KNOWLEDGE_DIR))))
    index_dir: Path = field(default_factory=lambda: Path(_str("INDEX_DIR", str(INDEX_DIR))))
    trace_dir: Path = field(default_factory=lambda: Path(_str("TRACE_DIR", str(TRACE_DIR))))
    workspace_dir: Path = field(default_factory=lambda: Path(_str("WORKSPACE_DIR", str(DATA_DIR / "workspace"))))

    # --- 派生属性 ---
    @property
    def resolved_chat_model(self) -> str:
        return self.chat_model or DEFAULT_CHAT_MODEL.get(self.provider, "mock-chat")

    @property
    def resolved_embed_model(self) -> str:
        return self.embed_model or DEFAULT_EMBED_MODEL.get(self.provider, "hash-ngram-256")

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or DEFAULT_BASE_URL.get(self.provider, "")

    @property
    def index_file(self) -> Path:
        return self.index_dir / "index.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """清空缓存后重新读取环境变量（测试与容器热更新用）。"""
    get_settings.cache_clear()
    return get_settings()
