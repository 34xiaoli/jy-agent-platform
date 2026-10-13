"""版面体检：量每一页的填充率、底部落点，并找出"孤儿标题"。

术语
----
孤儿标题（dangling heading）：小节标题出现在某页末尾，而它的正文全部在下一页。
这是简历 PDF 最刺眼的排版缺陷之一——读者会以为漏印了内容。

用法：
    python scripts/layout_check.py <pdf>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# 与 resume.html 中 .sec h2 的文字对应
HEADINGS = ("个人优势", "核心技能", "教育背景", "工作经历", "项目经历", "个人作品与开源", "自我评价")


def main() -> int:
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resume.pdf")
    document = fitz.open(pdf_path)
    print(f"文件：{pdf_path.name}｜页数：{document.page_count}\n")

    page_texts: list[str] = []
    issues: list[str] = []

    for page_no, page in enumerate(document, start=1):
        rect = page.rect
        blocks = [b for b in page.get_text("blocks") if b[4].strip()]
        if not blocks:
            issues.append(f"P{page_no} 空白页")
            continue
        top = min(b[1] for b in blocks)
        bottom = max(b[3] for b in blocks)
        usable_top, usable_bottom = 9 * 2.834, rect.height - 9 * 2.834  # @page margin 9mm
        fill = (bottom - usable_top) / (usable_bottom - usable_top) * 100
        print(f"第 {page_no} 页｜内容 {len(page.get_text('text'))} 字符")
        print(f"  纵向范围 {top:.0f}–{bottom:.0f}pt｜版心 {usable_top:.0f}–{usable_bottom:.0f}pt")
        print(f"  版心填充率 {fill:.1f}%｜距版心底 {usable_bottom - bottom:.0f}pt")
        page_texts.append(page.get_text("text"))

        # 页底 15% 区域的三种情形：留白 / 孤儿标题 / 正常正文
        threshold = usable_top + (usable_bottom - usable_top) * 0.85
        tail = [b for b in blocks if b[3] > threshold]
        tail_text = " ".join(b[4].strip() for b in tail).strip()
        if not tail_text:
            issues.append(f"P{page_no} 页底大片留白（末 15% 无任何内容）")
        elif tail_text in HEADINGS:
            issues.append(f"P{page_no} 孤儿标题：「{tail_text}」在页底，正文在下一页")
        elif len(tail_text) < 60 and fill > 95:
            issues.append(f"P{page_no} 页底疑似只剩零散文字（{len(tail_text)} 字符）：{tail_text[:40]}")

    # 相邻页内容量均衡度
    if len(page_texts) >= 2:
        sizes = [len(t) for t in page_texts]
        skew = (max(sizes) - min(sizes)) / max(sizes) * 100
        print(f"\n各页字符数 {sizes}｜最大偏差 {skew:.1f}%")
        if skew > 35:
            issues.append(f"两页内容量偏差 {skew:.1f}%（>35% 视为失衡）")

    print()
    if issues:
        print(f"发现 {len(issues)} 处版面问题：")
        for item in issues:
            print(f"  [!] {item}")
        return 1
    print("版面正常：无孤儿标题，页面填充均衡。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
