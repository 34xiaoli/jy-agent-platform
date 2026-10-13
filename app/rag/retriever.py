"""检索器：向量 + BM25 双路召回 → RRF 融合 → 重排 → 组装带引用的上下文。

这是整个 RAG 的"质量阀门"：
- 双路召回解决"字面精确（条款号、金额科目）"与"语义相近"两类问题；
- RRF（Reciprocal Rank Fusion）不依赖分数尺度，工程上比加权求和更稳；
- 重排负责把对的条款顶到第一位；
- 上下文组装强制带【引用标记】，让生成端"必须"依据证据。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.llm.base import LLMClient
from app.rag.chunker import Chunk
from app.rag.reranker import RankedChunk, heuristic_rerank, llm_rerank
from app.rag.store import LocalHybridIndex


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return self.chunk.citation

    def as_dict(self) -> dict[str, object]:
        return {
            "citation": self.chunk.citation,
            "doc_title": self.chunk.doc_title,
            "layer": self.chunk.layer,
            "authority": self.chunk.authority,
            "article": self.chunk.article,
            "score": self.score,
            "vector_score": round(self.vector_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "text": self.chunk.text,
        }


@dataclass
class RetrievalResult:
    query: str
    items: list[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    debug: dict[str, object] = field(default_factory=dict)

    @property
    def citations(self) -> list[str]:
        return [item.citation for item in self.items]

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "context_chars": len(self.context),
            "items": [item.as_dict() for item in self.items],
            "debug": self.debug,
        }


def render_context(items: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """把检索结果渲染成带引用标记的上下文（受字符预算约束）。"""
    blocks: list[str] = []
    used = 0
    for item in items:
        chunk = item.chunk
        header = (
            f"【{chunk.citation}】{chunk.doc_title}"
            f"（层级 {chunk.layer}｜{chunk.authority}｜条款 {chunk.article}）"
        )
        block = f"{header}\n{chunk.text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


class Retriever:
    """混合检索器：与索引后端解耦，替换向量库不影响上层编排。"""

    def __init__(
        self,
        index: LocalHybridIndex,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        layers: list[str] | None = None,
        use_hybrid: bool | None = None,
        use_rerank: bool | None = None,
        rerank_method: str = "heuristic",
    ) -> RetrievalResult:
        top_k = top_k or self.settings.top_k
        use_hybrid = self.settings.use_hybrid if use_hybrid is None else use_hybrid
        use_rerank = self.settings.use_rerank if use_rerank is None else use_rerank
        recall_k = max(self.settings.recall_k, top_k)
        debug: dict[str, object] = {"hybrid": use_hybrid, "rerank": use_rerank}

        if len(self.index) == 0:
            return RetrievalResult(query=query, debug={**debug, "empty_index": True})

        query_vector = self.llm.embed([query])[0]
        vector_hits = self.index.vector_search(query_vector, top_k=recall_k, layers=layers)
        bm25_hits: list[tuple[int, float]] = []
        if use_hybrid:
            bm25_hits = self.index.bm25_search(query, top_k=recall_k, layers=layers)

        # ---- RRF 融合：不依赖两路分数的量纲，工程上比加权求和更稳 ----
        rrf_k = self.settings.rrf_k
        fusion: dict[int, float] = {}
        for rank, (index, _) in enumerate(vector_hits):
            fusion[index] = fusion.get(index, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (index, _) in enumerate(bm25_hits):
            fusion[index] = fusion.get(index, 0.0) + 1.0 / (rrf_k + rank + 1)

        vector_map = dict(vector_hits)
        bm25_map = dict(bm25_hits)
        candidates: list[tuple[Chunk, float, dict[str, float]]] = []
        for index, score in sorted(fusion.items(), key=lambda item: item[1], reverse=True)[:recall_k]:
            candidates.append(
                (
                    self.index.get(index),
                    score,
                    {
                        "vector_score": round(vector_map.get(index, 0.0), 6),
                        "bm25_score": round(bm25_map.get(index, 0.0), 6),
                    },
                )
            )
        debug["candidates"] = len(candidates)

        if use_rerank and candidates:
            if rerank_method == "llm":
                ranked: list[RankedChunk] = llm_rerank(query, candidates, self.llm, self.settings.layer_prior)
                debug["rerank_method"] = "llm"
            else:
                ranked = heuristic_rerank(query, candidates, self.settings.layer_prior)
                debug["rerank_method"] = "heuristic"
        else:
            ranked = [
                RankedChunk(chunk=chunk, score=fusion_score, detail=detail)
                for chunk, fusion_score, detail in candidates
            ]
            debug["rerank_method"] = "none"

        items = [
            RetrievedChunk(
                chunk=item.chunk,
                score=round(item.score, 6),
                vector_score=item.detail.get("vector_score", 0.0),
                bm25_score=item.detail.get("bm25_score", 0.0),
                rrf_score=round(item.detail.get("fusion", 0.0), 6),
                detail=item.detail,
            )
            for item in ranked[: max(top_k, 1) * 2]
        ]
        context = render_context(items[:top_k], self.settings.max_context_chars)
        debug["top_score"] = items[0].score if items else 0.0
        return RetrievalResult(query=query, items=items[: top_k * 2], context=context, debug=debug)

