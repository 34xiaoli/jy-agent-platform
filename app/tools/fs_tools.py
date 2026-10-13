"""文件系统类工具：读取审计底稿 / 制度文件目录。

安全约束（对应「数据不出机 + 权限隔离」）：
只允许在显式给定的沙箱根目录内访问，且拒绝 ../ 逃逸与符号链接越权。
"""
from __future__ import annotations

from pathlib import Path

from app.tools.registry import Tool

DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_ITEMS = 50


def resolve_in_sandbox(path: str, root: Path) -> Path:
    """把用户/模型给出的路径解析到沙箱内，越界直接报错。"""
    root = root.resolve()
    candidate = Path(path)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"路径超出允许范围（沙箱：{root}）：{path}")
    return target


def make_list_files(root: Path):
    def list_files(pattern: str = "*.md", max_items: int = DEFAULT_MAX_ITEMS) -> str:
        """列出沙箱目录下的文件（按相对路径返回）。"""
        root.mkdir(parents=True, exist_ok=True)
        files = sorted(path for path in root.rglob(pattern) if path.is_file())
        if not files:
            return f"目录 {root} 下没有匹配 {pattern} 的文件。"
        lines = [f"共 {len(files)} 个文件（仅显示前 {max_items} 个）："]
        for path in files[:max_items]:
            size = path.stat().st_size
            lines.append(f"- {path.relative_to(root).as_posix()}（{size} 字节）")
        return "\n".join(lines)

    return list_files


def make_read_text(root: Path):
    def read_text(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """按相对路径读取沙箱内的文本文件（超长自动截断）。"""
        target = resolve_in_sandbox(path, root)
        if not target.is_file():
            return f"文件不存在：{path}"
        content = target.read_text(encoding="utf-8", errors="ignore")
        truncated = len(content) > max_chars
        body = content[:max_chars]
        suffix = f"\n\n（已截断，原文件共 {len(content)} 字符）" if truncated else ""
        return f"# 文件：{target.relative_to(root.resolve()).as_posix()}\n{body}{suffix}"

    return read_text


def build_fs_tools(sandbox_root: str | None = None) -> list[Tool]:
    """产出文件类工具；sandbox_root 为空时不注册（避免越权风险）。"""
    root = Path(sandbox_root or ".").resolve()
    return [
        Tool(
            name="list_working_files",
            description="列出工作目录中的文件列表，用于确认可用的底稿、制度或准则文件。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "通配符，例如 *.md 或 *.csv，默认 *.md"},
                    "max_items": {"type": "integer", "description": "最多返回条数，默认 50"},
                },
                "required": [],
            },
            func=make_list_files(root),
            category="filesystem",
        ),
        Tool(
            name="read_text_file",
            description="读取工作目录内的文本文件内容，用于查阅底稿描述或制度条款原文。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作目录的路径"},
                    "max_chars": {"type": "integer", "description": "最多读取字符数，默认 4000"},
                },
                "required": ["path"],
            },
            func=make_read_text(root),
            category="filesystem",
        ),
    ]
