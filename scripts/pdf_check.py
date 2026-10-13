"""体检 PDF：提取文本并检查乱码/缺字/溢出，作为交付前的最后一道关。

用法：
    python scripts/pdf_check.py "<pdf 路径>"
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

# 允许出现的字符：中日韩文字、拉丁字母数字、常用中文标点与技术符号
ALLOWED_PUNCT = set(
    "　、。，．；：？！…—–～·・｜／（）〔〕【】《》「」『』"
    "“”‘’´｀　-–—_+=<>/\\|~^`"
    "×÷±≈≠≤≥→←↑↓⇒∞∑√≈°％‰§¶†‡©®™€£¥$#@%&*()[]{}\"'.,:;?!"
)

SUSPICIOUS_HINTS = ("锟斤拷", "ï¿½", "â€", "Ã©", "Ã¤", "�", "□", "▯")


def is_susicious_char(ch: str) -> str | None:
    """返回可疑原因，正常字符返回 None。只拦截真正会显示成乱码的东西。"""
    if ch.isspace() or ch.isalnum():
        return None
    if ch in ALLOWED_PUNCT:
        return None
    code = ord(ch)
    if code == 0xFFFD:
        return "替换符 U+FFFD（乱码）"
    if 0xE000 <= code <= 0xF8FF or 0xF0000 <= code <= 0x10FFFF:
        return f"私用区字符 U+{code:04X}"
    category = unicodedata.category(ch)
    if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
        return f"控制/未分配字符 U+{code:04X}"
    # 几何图形（▸ ▪ ☆）与常用符号属正常排版
    if category in {"Sm", "Sk", "So"}:
        return None
    return f"非常规字符 {ch!r} U+{code:04X} {unicodedata.name(ch, '?')}"


def check_headers_footers(page, page_no: int, total: int) -> list[str]:
    """检测浏览器打印页眉页脚（日期 / file:// 文件路径 / 页码）残留。"""
    issues: list[str] = []
    for line in page.get_text("text").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "file:///" in stripped or re.match(r"^[A-Za-z]:\\", stripped):
            issues.append(f"P{page_no} 残留打印页脚（文件路径）：{stripped[:60]}")
        if re.match(r"^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}", stripped):
            issues.append(f"P{page_no} 残留打印页脚（日期时间）：{stripped[:40]}")
        if re.match(r"^\d+/\d+$", stripped):
            issues.append(f"P{page_no} 残留打印页脚（页码）：{stripped}")
    return issues



def main() -> int:
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resume.pdf")
    if not pdf_path.exists():
        print(f"文件不存在：{pdf_path}")
        return 2

    document = fitz.open(pdf_path)
    print(f"文件：{pdf_path.name}｜页数：{document.page_count}")

    issues: list[str] = []
    for page_no, page in enumerate(document, start=1):
        text = page.get_text("text")
        print(f"\n{'=' * 60}\n第 {page_no} 页｜{len(text)} 字符\n{'=' * 60}")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            reasons = [r for r in (is_susicious_char(c) for c in line) if r]
            hints = [h for h in SUSPICIOUS_HINTS if h in line]
            flag = "⚠" if (reasons or hints) else " "
            print(f"{flag} [{line_no:>3}] {line}")
            if reasons:
                issues.append(f"P{page_no} L{line_no} 字符问题：{reasons[:4]} → {line[:40]}")
            if hints:
                issues.append(f"P{page_no} L{line_no} 疑似乱码片段 {hints} → {line[:40]}")

        # 版面检查：内容是否超出可打印区域
        rect = page.rect
        blocks = page.get_text("blocks")
        overflow = [b for b in blocks if b[2] > rect.width - 10 or b[3] > rect.height - 10]
        for block in overflow:
            issues.append(f"P{page_no} 内容可能溢出页面：x1={block[2]:.0f} y1={block[3]:.0f}（页面 {rect.width:.0f}x{rect.height:.0f}）")

        # 打印页眉页脚残留检查
        issues.extend(check_headers_footers(page, page_no, document.page_count))

    print(f"\n{'=' * 60}")
    if issues:
        print(f"发现 {len(issues)} 处问题：")
        for item in issues:
            print(f"  ⚠ {item}")
        return 1
    print("✅ 未发现乱码、缺字或版面溢出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
