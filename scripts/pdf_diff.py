"""逐字符比对：HTML 源文本 vs PDF 提取文本，定位被吞掉/错位/替换的内容。

为什么需要这个
--------------
肉眼看 PDF 只能发现"明显不对"的地方（页眉页脚、方块字），
但漏字、错字、断句错位这类问题肉眼极易放过。
本脚本先做与顺序无关的**字符清点**（确定性回答"有没有字被吞掉"），
再做顺序比对定位阅读顺序差异。

用法：
    python scripts/pdf_diff.py <resume.html> <resume.pdf>
"""
from __future__ import annotations

import difflib
import html as html_lib
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

DROP_TAGS = re.compile(r"<(script|style|title)[^>]*>.*?</\1>", re.S | re.I)
# 页脚与悬浮打印按钮：HTML 里是元素，但 @media print 下 display:none，PDF 不应包含
DROP_BLOCKS = re.compile(
    r"<(footer[^>]*class=[\"']foot[\"'][^>]*>.*?</footer>)"
    r"|(<button[^>]*class=[\"']fab[\"'][^>]*>.*?</button>)",
    re.S | re.I,
)
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def html_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    raw = DROP_BLOCKS.sub(" ", raw)
    raw = DROP_TAGS.sub(" ", raw)
    raw = TAG.sub(" ", raw)
    raw = html_lib.unescape(raw)
    return WS.sub(" ", raw).strip()


def pdf_text(path: Path) -> str:
    document = fitz.open(path)
    return WS.sub(" ", " ".join(page.get_text("text") for page in document)).strip()


def compact(text: str) -> str:
    """去掉所有空白：消除 b 标签边界空格与 PDF 换行断词空格造成的假差异。"""
    return re.sub(r"\s+", "", text)


def main() -> int:
    if len(sys.argv) < 3:
        print("用法：python scripts/pdf_diff.py <resume.html> <resume.pdf>")
        return 2
    html_path, pdf_path = Path(sys.argv[1]), Path(sys.argv[2])
    raw_source, raw_rendered = html_text(html_path), pdf_text(pdf_path)

    # 装饰性符号：项目符号与获奖星标在 PDF 中以图形呈现
    for symbol in ("▸", "☆"):
        raw_source = raw_source.replace(symbol, "")
        raw_rendered = raw_rendered.replace(symbol, "")

    # 页脚在 HTML 中 display:none，若混进 PDF 即为打印缺陷
    footer = compact("AI智能体（Agent）开发工程师·个人简历")

    source = compact(raw_source)
    rendered = compact(raw_rendered)
    footer_leaked = footer in rendered

    print(f"HTML 纯文本 {len(raw_source)} 字符（去空白 {len(source)}）")
    print(f"PDF  提取   {len(raw_rendered)} 字符（去空白 {len(rendered)}）")

    # 1) 字符清点：与顺序无关，确定性回答"有没有字被吞掉或替换"
    delta = Counter(source) - Counter(rendered)
    extra = Counter(rendered) - Counter(source)
    print(f"\n[字符清点] HTML 侧多出 {sum(delta.values())}；PDF 侧多出 {sum(extra.values())}")
    if delta or extra:
        for char, count in delta.items():
            print(f"  [!] PDF 缺少 {count} 个 {char!r} U+{ord(char):04X} {unicodedata.name(char, '?')}")
        for char, count in extra.items():
            print(f"  [!] PDF 多出 {count} 个 {char!r} U+{ord(char):04X} {unicodedata.name(char, '?')}")
    else:
        print("  OK 字符完全一致：没有任何文字被吞掉、替换或重复。")

    # 2) 顺序比对：定位阅读顺序差异（分页与页眉位置属正常排版行为）
    matcher = difflib.SequenceMatcher(None, source, rendered, autojunk=False)
    print(f"\n[顺序比对] 相似度 {matcher.ratio():.4f}（跨页断句与页眉位置属正常）")
    moves = [
        (tag, source[i1:i2], rendered[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal" and (i2 - i1 > 4 or j2 - j1 > 4)
    ]
    for tag, missing, extra_text in moves[:12]:
        print(f"  - [{tag}] HTML:{missing[:46] or '(无)'} | PDF:{extra_text[:46] or '(无)'}")
    if len(moves) > 12:
        print(f"  - 另有 {len(moves) - 12} 处（多为跨页断句）")

    if footer_leaked:
        print(f"\n  [!] PDF 中混入了页脚：{footer}")
    return 0 if (not delta and not extra and not footer_leaked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
