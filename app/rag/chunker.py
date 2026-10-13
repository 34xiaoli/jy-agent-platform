"""结构化切片：按「条款 / 小节」切，而不是按固定字数切。

为什么不能用固定长度切片：
审计与合规问答依赖「条款号」这类精确坐标。把相邻两个条款切进同一个块，
会造成「检索到了、但引用张冠李戴」——这是准则引用最典型的失败模式。

本模块的做法：
1. 解析 front-matter 元数据（准则编号、层级、权威级别、生效日期）；
2. 按「第X条 / 第X章 / （一） / 1.1 / 一、」等结构标记切块；
3. 文档级元数据下沉到每个块，并生成强制引用标记（如 L2·CSA1313·第13条）；
4. 超长条款按句子边界二次切分（保留父条款号）；过短块向上合并，避免碎片。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

MAX_CHARS = 600
MIN_CHARS = 50

HEADING_PATTERNS = (
    re.compile(r"^第\s*[一二三四五六七八九十百零〇\d]+\s*[条章节]"),
    re.compile(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[)）]"),
    re.compile(r"^\d+(\.\d+)+\s*"),
    re.compile(r"^[一二三四五六七八九十]+\s*、"),
)
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。；！？])")
MARKUP_RE = re.compile(r"[*`>#]")
ARTICLE_LABEL_RE = re.compile(
    r"^(第\s*[一二三四五六七八九十百零〇\d]+\s*[条章节]"
    r"|[（(]\s*[一二三四五六七八九十\d]+\s*[)）]"
    r"|\d+(?:\.\d+)+"
    r"|[一二三四五六七八九十]+\s*、)"
)

AUTHORITY_BY_LAYER = {
    "L1": "法律法规",
    "L2": "执业准则",
    "L3": "实务经验",
    "L4": "内部数据",
}


@dataclass
class Chunk:
    """检索的最小单元：文本 + 可追溯元数据 + 强制引用标记。"""

    chunk_id: str
    text: str
    layer: str
    doc_id: str
    doc_title: str
    article: str = "全文"
    authority: str = ""
    effective_date: str = ""
    source_path: str = ""
    citation: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip()
    return meta, raw[match.end() :]


def is_heading(line: str) -> bool:
    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def extract_article_label(line: str) -> str:
    """从标题行中只取条款标识（如「第 2 条」「（一）」「3.1」），避免把正文带进引用标记。"""
    match = ARTICLE_LABEL_RE.match(line.strip())
    label = match.group(1) if match else line.strip()[:40]
    return re.sub(r"\s+", "", label)


def clean_text(text: str) -> str:
    lines = [MARKUP_RE.sub("", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def split_long_block(block: str, max_chars: int = MAX_CHARS) -> list[str]:
    """按句子边界把超长条款切开，保留语义完整（不切断句子）。"""
    if len(block) <= max_chars:
        return [block]
    pieces: list[str] = []
    buffer = ""
    for sentence in SENTENCE_SPLIT_RE.split(block):
        if not sentence:
            continue
        if len(buffer) + len(sentence) > max_chars and buffer:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer += sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def split_document(
    raw: str,
    layer: str,
    source_path: str = "",
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Chunk]:
    """把一份文档切分成结构化块（带完整元数据与引用标记）。"""
    layer = layer.upper()
    meta, body = parse_front_matter(raw)
    doc_id = meta.get("doc_id") or Path(source_path).stem or "DOC"
    doc_title = meta.get("title") or doc_id
    authority = meta.get("authority") or AUTHORITY_BY_LAYER.get(layer, "")
    effective_date = meta.get("effective_date", "")

    blocks: list[tuple[str, str]] = []  # (article, text)
    current_article = "全文"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = clean_text("\n".join(buffer))
            if text:
                blocks.append((current_article, text))

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = stripped.lstrip("#").strip()  # Markdown 标题符要先剥离，否则结构标记识别不到
        if is_heading(normalized):
            flush()
            buffer.clear()
            current_article = extract_article_label(normalized)
        buffer.append(stripped)
    flush()

    # 过短块先向前合并（文档开头的纯标题），其余向上合并，避免产生"只有标题"的空块
    merged: list[tuple[str, str]] = []
    pending = ""
    for article, text in blocks:
        if len(text) < min_chars:
            if merged:
                prev_article, prev_text = merged[-1]
                merged[-1] = (prev_article, f"{prev_text}\n{text}")
            else:
                pending = f"{pending}\n{text}".strip()
            continue
        merged.append((article, f"{pending}\n{text}".strip() if pending else text))
        pending = ""
    if pending and merged:
        prev_article, prev_text = merged[-1]
        merged[-1] = (prev_article, f"{prev_text}\n{pending}")

    chunks: list[Chunk] = []
    for article, text in merged:
        pieces = split_long_block(text, max_chars)
        for index, piece in enumerate(pieces, start=1):
            suffix = "" if len(pieces) == 1 else f"（{index}）"
            citation = build_citation(layer, doc_id, article)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks):03d}",
                    text=piece,
                    layer=layer,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    article=article,
                    authority=authority,
                    effective_date=effective_date,
                    source_path=source_path,
                    citation=f"{citation}{suffix}",
                )
            )
    return chunks


def build_citation(layer: str, doc_id: str, article: str) -> str:
    """生成强制引用标记，例如 L2·CSA1313·第13条。"""
    article = article.replace("\n", " ").strip() or "全文"
    return f"{layer}·{doc_id}·{article}"


def load_documents(knowledge_dir: Path) -> list[Chunk]:
    """加载 data/knowledge/L1..L4 下的所有 .md，按目录名决定层级。"""
    chunks: list[Chunk] = []
    if not knowledge_dir.exists():
        return chunks
    for path in sorted(knowledge_dir.rglob("*.md")):
        layer = path.parent.name.upper()
        if not re.fullmatch(r"L[1-4]", layer):
            continue
        raw = path.read_text(encoding="utf-8")
        chunks.extend(split_document(raw, layer=layer, source_path=str(path)))
    return chunks


def iter_chunk_dicts(chunks: Iterable[Chunk]) -> list[dict[str, Any]]:
    return [chunk.as_dict() for chunk in chunks]
