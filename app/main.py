"""对外 HTTP 能力服务层（五层架构的第 4 层）。

启动：
    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

接口：
    GET  /health        服务 / 推理后端 / 索引健康状态
    POST /chat          主智能体问答（返回结论 + 引用 + 校验结果 + trace_id）
    POST /retrieve      仅检索（调试与评测用，可对比 hybrid / rerank 开关）
    GET  /tools         已注册工具清单（function calling schema）
    POST /ingest        重建索引
    GET  /metrics       基于 trace 的运行指标
    GET  /chat/stream   SSE 流式输出

治理能力：
    - 可选 X-API-Key 鉴权（SERVICE_API_KEY）
    - 每 IP 每分钟限流（RATE_LIMIT_PER_MIN）
    - 角色（role）在检索入口决定可见知识层级，权限隔离不依赖提示词
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.agents.orchestrator import Orchestrator
from app.config import Settings, get_settings
from app.knowledge.layers import allowed_layers, summarize_index
from app.llm.base import LLMError
from app.llm.factory import build_llm
from app.obs.tracer import summarize_records
from app.rag.retriever import Retriever
from app.rag.store import LocalHybridIndex
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.tools.registry import ToolRegistry, build_default_registry


class Services:
    """进程内单例容器：启动时构建一次，避免每个请求重建索引与客户端。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = build_llm(settings)
        self.registry: ToolRegistry = build_default_registry(str(settings.workspace_dir))
        self.index = self._load_or_build_index()
        self.retriever = Retriever(self.index, self.llm, settings)
        self.orchestrator = Orchestrator(
            llm=self.llm, retriever=self.retriever, registry=self.registry, settings=settings
        )

    def _load_or_build_index(self) -> LocalHybridIndex:
        if self.settings.index_file.exists():
            return LocalHybridIndex.load(self.settings.index_file)
        from scripts.ingest import build_index

        index, _, _ = build_index(self.settings, verbose=False)
        return index

    def rebuild(self) -> tuple[int, dict[str, int], float]:
        from scripts.ingest import build_index

        started = time.perf_counter()
        index, _, summary = build_index(self.settings, verbose=False)
        self.index = index
        self.retriever.index = index
        self.orchestrator.retriever = self.retriever
        return len(index), summary, (time.perf_counter() - started) * 1000


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.services = Services(settings)
        yield

    application = FastAPI(
        title="审计合规 Agent 平台 · 能力服务层",
        version=__version__,
        description="分层知识库（L1–L4）+ 混合检索（BM25/向量 + RRF + 重排）+ 多智能体编排 + 引用校验",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.hits = defaultdict(deque)

    @application.middleware("http")
    async def rate_limit(request: Request, call_next: Callable) -> Any:
        """极简滑动窗口限流：内网场景够用；分布式部署应换成 Redis 计数器。"""
        if settings.rate_limit_per_min <= 0 or request.url.path == "/health":
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        window: deque[float] = application.state.hits[client]
        now = time.time()
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= settings.rate_limit_per_min:
            return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": "请求过于频繁"})
        window.append(now)
        return await call_next(request)

    _install_routes(application, settings)
    return application


def services(request: Request) -> Services:
    return request.app.state.services


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = get_settings().service_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key 无效或缺失")


def _to_citations(items: list[Any]) -> list[Citation]:
    return [
        Citation(
            citation=item.citation,
            doc_title=item.chunk.doc_title,
            layer=item.chunk.layer,
            authority=item.chunk.authority,
            article=item.chunk.article,
            score=item.score,
            vector_score=item.vector_score,
            bm25_score=item.bm25_score,
            text=item.chunk.text,
        )
        for item in items
    ]


def _install_routes(application: FastAPI, settings: Settings) -> None:
    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(container: Services = Depends(services)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            provider=settings.provider,
            chat_model=settings.resolved_chat_model,
            embed_model=settings.resolved_embed_model,
            index_chunks=len(container.index),
            layers=summarize_index(container.index.chunks),
            llm=container.llm.health(),
            tools=container.registry.names(),
            langfuse_enabled=settings.enable_langfuse,
        )

    @application.post("/chat", response_model=ChatResponse, tags=["agent"], dependencies=[Depends(require_api_key)])
    def chat(payload: ChatRequest, container: Services = Depends(services)) -> ChatResponse:
        started = time.perf_counter()
        try:
            state, record = container.orchestrator.answer(payload.question, role=payload.role, top_k=payload.top_k)
        except LLMError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return ChatResponse(
            question=payload.question,
            role=payload.role,
            intent=state.intent,
            answer=state.answer,
            verified=state.verified,
            refused=state.refused,
            citations=state.citations,
            verification_notes=state.verification_notes,
            evidence=_to_citations(state.retrieval.items if state.retrieval else []),
            plan=[step.as_dict() for step in state.steps],
            usage=state.usage,
            trace_id=str(record.get("trace_id", "")),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @application.post("/retrieve", response_model=RetrieveResponse, tags=["agent"])
    def retrieve(payload: RetrieveRequest, container: Services = Depends(services)) -> RetrieveResponse:
        result = container.retriever.retrieve(
            payload.query,
            top_k=payload.top_k,
            layers=allowed_layers(payload.role),
            use_hybrid=payload.use_hybrid,
            use_rerank=payload.use_rerank,
            rerank_method=payload.rerank_method,
        )
        return RetrieveResponse(
            query=result.query,
            context_chars=len(result.context),
            items=_to_citations(result.items),
            debug=dict(result.debug),
        )

    @application.get("/tools", tags=["agent"])
    def tools(container: Services = Depends(services)) -> dict[str, Any]:
        return {"tools": container.registry.specs()}

    @application.post("/ingest", response_model=IngestResponse, tags=["ops"], dependencies=[Depends(require_api_key)])
    def ingest(container: Services = Depends(services)) -> IngestResponse:
        chunks, summary, duration_ms = container.rebuild()
        return IngestResponse(
            chunks=chunks,
            layers=summary,
            index_file=str(settings.index_file),
            duration_ms=int(duration_ms),
        )

    @application.get("/metrics", tags=["ops"])
    def metrics(container: Services = Depends(services)) -> dict[str, Any]:
        trace_file = settings.trace_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        return {"traces": summarize_records(trace_file), "index_chunks": len(container.index)}

    @application.get("/chat/stream", tags=["agent"])
    def chat_stream(question: str, role: str = "assistant") -> StreamingResponse:
        """SSE 流式输出：先推 trace 与引用，再逐行推正文（改善长任务体感延迟）。"""

        def event_stream() -> Any:
            container: Services = application.state.services
            state, record = container.orchestrator.answer(question, role=role)
            yield f"event: meta\ndata: {record.get('trace_id', '')}\n\n"
            for citation in state.citations:
                yield f"event: citation\ndata: {citation}\n\n"
            for line in state.answer.splitlines():
                yield f"event: token\ndata: {line}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")


app = create_app()


