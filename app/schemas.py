"""对外 API 的数据契约（Pydantic 模型）。

独立成文件的原因：能力服务层的入参出参需要被 API、评测脚本、外部系统三方复用，
把契约集中定义可以避免"实现改了、调用方不知道"。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["intern", "assistant", "manager", "partner"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="用户问题")
    role: Role = Field("assistant", description="用户角色，决定可见知识层级（L1–L4）")
    top_k: int = Field(5, ge=1, le=20, description="返回的证据条数")
    rerank_method: Literal["heuristic", "llm"] = Field("heuristic", description="重排策略")
    use_hybrid: bool = Field(True, description="是否启用 BM25 + 向量混合召回")
    use_rerank: bool = Field(True, description="是否启用重排")


class Citation(BaseModel):
    citation: str = Field(..., description="强制引用标记，如 L2·CSA1313·第6条")
    doc_title: str = ""
    layer: str = ""
    authority: str = ""
    article: str = ""
    score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    text: str = ""


class ChatResponse(BaseModel):
    question: str
    role: str
    intent: str
    answer: str
    verified: bool = Field(..., description="引用校验是否通过")
    refused: bool = Field(..., description="是否因证据不足而按规则拒答")
    citations: list[str] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)
    evidence: list[Citation] = Field(default_factory=list)
    plan: list[dict[str, str]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    trace_id: str = ""
    latency_ms: int = 0


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    role: Role = "assistant"
    top_k: int = Field(5, ge=1, le=20)
    rerank_method: Literal["heuristic", "llm"] = "heuristic"
    use_hybrid: bool = Field(True, description="是否启用 BM25 + 向量混合召回")
    use_rerank: bool = Field(True, description="是否启用重排")


class RetrieveResponse(BaseModel):
    query: str
    context_chars: int
    items: list[Citation]
    debug: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    chat_model: str
    embed_model: str
    index_chunks: int
    layers: dict[str, int]
    llm: dict[str, Any]
    tools: list[str]
    langfuse_enabled: bool


class IngestResponse(BaseModel):
    chunks: int
    layers: dict[str, int]
    index_file: str
    duration_ms: int
