"""重排（Rerank）：把「能召回」变成「排得准」。

三级策略，按可用资源切换：
1. heuristic（默认，零依赖）：词项覆盖率 + 条款号精确命中 + 准则编号命中 + 层级先验；
2. llm（需模型）：listwise 让模型在候选里排序（候选 ≤ 20 时效果明显）；
3. cross-encoder（扩展位）：BGE-reranker-v2-m3 / Qwen3-Reranker，工程上只需实现同一签名。

为什么合规场景必须有重排：向量召回偏语义、BM25 偏字面，融合后的 top-k 仍会
混入「语义相近但条款不对」的块。审计问答里「排第一的条款是不是对的那一条」
是硬指标，靠召回不够，必须精排。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from app.config import get_settings
from app.llm.base import LLMClient
from app.rag.chunker import Chunk
from app.rag.store import tokenize

ARTICLE_IN_QUERY_RE = re.compile(r"第\s*[一二三四五六七八九十百零〇\d]+\s*[条章节]")
DOC_CODE_RE = re.compile(r"[A-Za-z]{2,}\s?\d{2,4}")
NUMBER_RE = re.compile(r"\d{2,4}")

WEIGHT_COVERAGE = 0.45
WEIGHT_ARTICLE = 0.20
WEIGHT_DOC = 0.15
WEIGHT_PRIOR = 0.20
FUSION_WEIGHT = 0.30  # 融合分在最终分中的占比


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float
    detail: dict[str, float] = field(default_factory=dict)


def _article_hit(query: str, chunk: Chunk) -> float:
    wanted = ARTICLE_IN_QUERY_RE.findall(query)
    if not wanted:
        return 0.0
    normalized = chunk.article.replace(" ", "")
    return 1.0 if any(item.replace(" ", "") in normalized for item in wanted) else 0.0


def _doc_hit(query: str, chunk: Chunk) -> float:
    codes = [match.replace(" ", "").upper() for match in DOC_CODE_RE.findall(query)]
    numbers = NUMBER_RE.findall(query)
    doc_id = chunk.doc_id.replace(" ", "").upper()
    if any(code in doc_id or doc_id in code for code in codes if len(code) >= 4):
        return 1.0
    if any(number in doc_id for number in numbers if len(number) >= 4):
        return 1.0
    return 0.0


def heuristic_rerank(
    query: str,
    candidates: Sequence[tuple[Chunk, float, dict[str, float]]],
    layer_prior: dict[str, float] | None = None,
) -> list[RankedChunk]:
    """零依赖启发式重排。candidates 元素为 (chunk, 融合分, 明细)。"""
    prior_map = layer_prior or get_settings().layer_prior
    query_tokens = set(tokenize(query))
    max_fusion = max((fusion for _, fusion, _ in candidates), default=1.0) or 1.0

    ranked: list[RankedChunk] = []
    for chunk, fusion, detail in candidates:
        chunk_tokens = set(tokenize(chunk.text + " " + chunk.article + " " + chunk.doc_title))
        coverage = len(query_tokens & chunk_tokens) / len(query_tokens) if query_tokens else 0.0
        article = _article_hit(query, chunk)
        doc = _doc_hit(query, chunk)
        prior = prior_map.get(chunk.layer, 0.85)
        feature_score = (
            WEIGHT_COVERAGE * coverage
            + WEIGHT_ARTICLE * article
            + WEIGHT_DOC * doc
            + WEIGHT_PRIOR * prior
        )
        score = (1 - FUSION_WEIGHT) * feature_score + FUSION_WEIGHT * (fusion / max_fusion)
        ranked.append(
            RankedChunk(
                chunk=chunk,
                score=round(score, 6),
                detail={
                    "coverage": round(coverage, 6),
                    "article_hit": article,
                    "doc_hit": doc,
                    "layer_prior": prior,
                    "fusion": round(fusion, 6),
                    **detail,
                },
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


LLM_RERANK_PROMPT = """你是审计准则检索结果的相关性评审员。
用户问题：{query}

候选条款：
{candidates}

请只输出 JSON：{{"ranking": [按相关性从高到低排列的候选序号]}}
只排序，不要解释，不要输出其他内容。"""


def llm_rerank(
    query: str,
    candidates: Sequence[tuple[Chunk, float, dict[str, float]]],
    llm: LLMClient,
    layer_prior: dict[str, float] | None = None,
) -> list[RankedChunk]:
    """listwise 重排：失败自动回退到启发式，保证主链路不中断。"""
    fallback = heuristic_rerank(query, candidates, layer_prior)
    if len(candidates) < 2:
        return fallback
    listing = "\n".join(
        f"{index + 1}. 【{chunk.citation}】{chunk.text[:160]}"
        for index, (chunk, _, _) in enumerate(candidates)
    )
    try:
        result = llm.chat(
            [{"role": "user", "content": LLM_RERANK_PROMPT.format(query=query, candidates=listing)}],
            temperature=0.0,
            max_tokens=256,
        )
        payload = json.loads(re.search(r"\{.*\}", result.text, re.S).group(0))
        order = [int(item) for item in payload.get("ranking", [])]
    except Exception:  # noqa: BLE001 - 重排失败绝不能影响主链路
        return fallback

    ranked: list[RankedChunk] = []
    seen: set[int] = set()
    for position, one_based in enumerate(order):
        index = one_based - 1
        if not 0 <= index < len(candidates) or index in seen:
            continue
        seen.add(index)
        chunk, fusion, detail = candidates[index]
        ranked.append(
            RankedChunk(
                chunk=chunk,
                score=round(1.0 - position * 0.01, 6),
                detail={"llm_rank": position + 1, "fusion": round(fusion, 6), **detail},
            )
        )
    for index, (chunk, fusion, detail) in enumerate(candidates):
        if index not in seen:
            ranked.append(
                RankedChunk(chunk=chunk, score=0.0, detail={"llm_rank": 0, "fusion": fusion, **detail})
            )
    return ranked
