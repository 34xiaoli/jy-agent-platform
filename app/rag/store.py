"""本地混合索引（向量 + BM25），纯标准库实现，零依赖即可运行。

生产规模（百万级 / 多租户）应替换为 Milvus / Qdrant / PGVector / Elasticsearch。
本模块刻意保持与上层解耦：Retriever 只调用下面这几个方法，
因此替换后端时不需要改动编排与提示词逻辑。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.rag.chunker import Chunk

K1 = 1.5
B = 0.75
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ALNUM_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文/数字按词，中文按字 + 二元组（无需外部分词器）。"""
    lowered = text.lower()
    tokens = ALNUM_RE.findall(lowered)
    cjk = CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))
    return tokens or ["<empty>"]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class LocalHybridIndex:
    """内存索引 + JSON 持久化，支持向量检索与 BM25 词法检索。"""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._token_counts: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._lengths: list[int] = []

    # ------------------------------------------------------------------ 写入
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 数量不一致：{len(chunks)} vs {len(vectors)}")
        for chunk, vector in zip(chunks, vectors):
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self.chunks.append(chunk)
            self.vectors.append([float(value) for value in vector])
            self._token_counts.append(counts)
            self._lengths.append(len(tokens))
            for token in counts:
                self._doc_freq[token] += 1

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ 检索
    def _allowed(self, index: int, layers: set[str] | None) -> bool:
        return layers is None or self.chunks[index].layer in layers

    def vector_search(
        self, query_vector: Sequence[float], top_k: int = 20, layers: Iterable[str] | None = None
    ) -> list[tuple[int, float]]:
        layer_set = {layer.upper() for layer in layers} if layers else None
        scored = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
            if self._allowed(index, layer_set)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def bm25_search(
        self, query: str, top_k: int = 20, layers: Iterable[str] | None = None
    ) -> list[tuple[int, float]]:
        layer_set = {layer.upper() for layer in layers} if layers else None
        total = len(self.chunks) or 1
        avg_len = (sum(self._lengths) / total) if self._lengths else 1.0
        query_tokens = set(tokenize(query))
        scored: list[tuple[int, float]] = []
        for index, counts in enumerate(self._token_counts):
            if not self._allowed(index, layer_set):
                continue
            length = self._lengths[index] or 1
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if frequency == 0:
                    continue
                document_freq = self._doc_freq.get(token, 0) or 1
                idf = math.log(1 + (total - document_freq + 0.5) / (document_freq + 0.5))
                denominator = frequency + K1 * (1 - B + B * length / avg_len)
                score += idf * (frequency * (K1 + 1)) / denominator
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get(self, index: int) -> Chunk:
        return self.chunks[index]

    # -------------------------------------------------------------- 持久化
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
            "vectors": [[round(value, 6) for value in vector] for vector in self.vectors],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalHybridIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls()
        chunks = []
        for item in payload.get("chunks", []):
            data = dict(item)
            data.setdefault("extra", {})
            chunks.append(Chunk(**data))
        index.add(chunks, payload.get("vectors", []))
        return index
