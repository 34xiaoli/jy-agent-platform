"""建索引脚本：读取 data/knowledge/L1–L4 → 结构化切片 → 向量化 → 落盘。

用法：
    python scripts/ingest.py
    python scripts/ingest.py --knowledge-dir data/knowledge --index-file data/index/index.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings, get_settings  # noqa: E402
from app.knowledge.layers import summarize_index  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402
from app.rag.chunker import load_documents  # noqa: E402
from app.rag.store import LocalHybridIndex  # noqa: E402

BATCH_SIZE = 16


def embed_in_batches(llm, texts: list[str], batch_size: int = BATCH_SIZE, verbose: bool = True) -> list[list[float]]:
    """分批向量化：避免一次性大请求导致显存或超时问题。"""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        window = texts[start : start + batch_size]
        vectors.extend(llm.embed(window))
        if verbose:
            print(f"  已向量化 {min(start + batch_size, len(texts))}/{len(texts)}")
    return vectors


def build_index(
    settings: Settings | None = None,
    knowledge_dir: str | None = None,
    index_file: str | None = None,
    verbose: bool = True,
) -> tuple[LocalHybridIndex, Path, dict[str, int]]:
    """构建并持久化索引，返回 (索引, 索引文件路径, 各层块数统计)。"""
    settings = settings or get_settings()
    kdir = Path(knowledge_dir) if knowledge_dir else settings.knowledge_dir
    ifile = Path(index_file) if index_file else settings.index_file

    chunks = load_documents(kdir)
    if not chunks:
        raise SystemExit(f"未在 {kdir} 找到任何 .md 文件（目录结构应为 L1/L2/L3/L4）")

    llm = build_llm(settings)
    if verbose:
        print(f"[ingest] 切片完成：{len(chunks)} 块；开始向量化（{settings.resolved_embed_model}）")
    vectors = embed_in_batches(llm, [chunk.text for chunk in chunks], verbose=verbose)

    index = LocalHybridIndex()
    index.add(chunks, vectors)
    index.save(ifile)
    summary = summarize_index(chunks)
    if verbose:
        print(f"[ingest] 各层块数：{summary}")
        print(f"[ingest] 索引已保存：{ifile}")
    return index, ifile, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 L1–L4 分层知识库索引")
    parser.add_argument("--knowledge-dir", default=None, help="知识库根目录（默认 data/knowledge）")
    parser.add_argument("--index-file", default=None, help="索引输出路径（默认 data/index/index.json）")
    args = parser.parse_args()

    settings = get_settings()
    started = time.perf_counter()
    print(
        f"[ingest] provider={settings.provider} chat={settings.resolved_chat_model} "
        f"embed={settings.resolved_embed_model}"
    )
    _, ifile, summary = build_index(settings, args.knowledge_dir, args.index_file)
    print(f"[ingest] 完成，耗时 {time.perf_counter() - started:.2f}s；层级统计 {summary}；输出 {ifile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
