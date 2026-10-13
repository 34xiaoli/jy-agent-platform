"""MCP Server：把审计知识库与确定性工具暴露为标准 MCP 协议（stdio + JSON-RPC 2.0）。

为什么做这个
------------
MCP（Model Context Protocol）已成为「模型 / Agent 接入外部工具与数据源」的事实标准。
把审计检索、底稿读取、重要性水平与抽样计算做成 MCP Server 后，
同一套工具可被任意支持 MCP 的客户端复用，不必为每个客户端重写集成代码。

实现取舍
--------
1. 零三方依赖：手写 MCP 方法子集（initialize / ping / tools/list / tools/call），
   只用标准库，便于内网 / 无网环境验证协议正确性，也便于逐行讲清原理；
   换成官方 `mcp` SDK 时，下面的 TOOL_SPECS 与 call_tool 逻辑可直接复用。
2. 懒加载：initialize 只回协议信息，索引与检索器在首次 tools/call 时才构建，
   缩短客户端握手等待。
3. 只读 + 沙箱：所有工具均为只读或纯计算，不提供写文件 / 执行命令能力。
4. 日志走 stderr：stdout 只能输出 JSON-RPC 报文，否则会污染协议流。

用法：
    python -m mcp.audit_mcp_server
客户端配置（示例）：
    {
      "mcpServers": {
        "audit-compliance": {
          "command": "python",
          "args": ["-m", "mcp.audit_mcp_server"],
          "env": {"LLM_PROVIDER": "mock"}
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.layers import allowed_layers  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.store import LocalHybridIndex  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402

SERVER_NAME = "jy-audit-compliance"
SERVER_VERSION = "0.1.0"
# 由新到旧排列：握手时回落到客户端请求的版本（若支持），否则用最新
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_audit_knowledge",
        "description": (
            "在 L1 法规 / L2 执业准则 / L3 实务经验 / L4 内部数据四层知识库中做混合检索"
            "（向量 + BM25 双路召回 → RRF 融合 → 重排），返回带条款级引用的原文片段。"
            "回答准则、程序、抽样、监管要求类问题时优先调用本工具，不要凭记忆作答。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言问题或准则编号关键词，如「函证控制在什么情况下必须执行」"},
                "role": {
                    "type": "string",
                    "enum": ["intern", "assistant", "manager", "partner"],
                    "description": "调用者角色，决定可见知识层级（L4 内部数据仅 manager/partner 可见）；默认 assistant",
                },
                "top_k": {"type": "integer", "description": "返回片段数，默认 5，最大 20"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_workpaper",
        "description": "读取工作目录（沙箱）内的审计底稿或制度文件文本，用于核对底稿描述与原文条款。禁止访问沙箱之外的路径。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作目录的路径，如「底稿-A2026-应收账款-函证-01.md」"},
                "max_chars": {"type": "integer", "description": "最多读取字符数，默认 4000"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_workpapers",
        "description": "列出工作目录内的文件清单，用于确认有哪些底稿 / 制度 / 准则原文可读。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "通配符，默认 *.md"},
                "max_items": {"type": "integer", "description": "最多返回条数，默认 50"},
            },
            "required": [],
        },
    },
    {
        "name": "calc_materiality",
        "description": "计算财务报表整体重要性水平、实际执行重要性与明显微小错报临界值。确定性计算，必须用本工具而非模型心算。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "benchmark": {"type": "number", "description": "基准金额，如营业收入 / 资产总额 / 税前利润，须大于 0"},
                "rate": {"type": "number", "description": "比例，默认 0.05 表示 5%（惯例 0.5%–5%）"},
            },
            "required": ["benchmark"],
        },
    },
    {
        "name": "monetary_unit_sampling",
        "description": "货币单元抽样（PPS）：按金额大小抽取样本，输出抽样间隔、必选高价值项目与选中索引。确定性计算。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amounts": {"type": "array", "items": {"type": "number"}, "description": "总体各项目金额"},
                "sample_size": {"type": "integer", "description": "目标样本量，须为正整数"},
            },
            "required": ["amounts", "sample_size"],
        },
    },
]


def log(message: str) -> None:
    """日志必须走 stderr：stdout 是 JSON-RPC 专用通道。"""
    print(f"[mcp] {message}", file=sys.stderr, flush=True)


class AuditMcpServer:
    """把仓库内已有的检索与工具能力直接复用为 MCP 工具（不重复实现业务逻辑）。"""

    def __init__(self) -> None:
        self._retriever: Retriever | None = None
        self._registry = None
        self._settings = None

    def _ensure(self) -> None:
        if self._retriever is not None:
            return
        self._settings = get_settings()
        self._registry = build_default_registry(str(self._settings.workspace_dir))
        llm = build_llm(self._settings)
        index_file = self._settings.index_file
        if index_file.exists():
            index = LocalHybridIndex.load(index_file)
        else:  # 首次启动且未建索引：现场构建，保证 server 永远可用
            from scripts.ingest import build_index

            log("未发现索引文件，正在构建……")
            index, _, _ = build_index(self._settings, verbose=False)
        self._retriever = Retriever(index, llm, self._settings)
        log(f"就绪：provider={self._settings.provider}，索引 {len(index)} 块")

    def _search_audit_knowledge(self, args: dict[str, Any]) -> str:
        self._ensure()
        role = str(args.get("role") or "assistant")
        top_k = max(1, min(int(args.get("top_k") or 5), 20))
        result = self._retriever.retrieve(  # type: ignore[union-attr]
            str(args["query"]), top_k=top_k, layers=allowed_layers(role)
        )
        if not result.items:
            return "未召回到任何依据条款。按规则应当明确告知用户「知识库中无对应依据」，不要自由发挥。"
        lines = [f"命中 {len(result.items)} 条依据（角色 {role}，可见层级 {allowed_layers(role)}）：", ""]
        for position, item in enumerate(result.items, start=1):
            chunk = item.chunk
            lines.append(
                f"[{position}] 【{chunk.citation}】{chunk.doc_title}"
                f"（层级 {chunk.layer}｜权威性 {chunk.authority}｜得分 {item.score:.4f}）"
            )
            lines.append(chunk.text)
            lines.append("")
        lines.append("作答要求：只依据以上条款作答并标注引用编号；依据不足时明确说明缺口，不要补全。")
        return "\n".join(lines)

    def _call_registry_tool(self, name: str, args: dict[str, Any]) -> str:
        self._ensure()
        result = self._registry.call(name, args)  # type: ignore[union-attr]
        if not result.ok:
            raise RuntimeError(result.error or "工具执行失败")
        return result.output

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        handlers: dict[str, Callable[[], str]] = {
            "search_audit_knowledge": lambda: self._search_audit_knowledge(arguments),
            "read_workpaper": lambda: self._call_registry_tool("read_text_file", arguments),
            "list_workpapers": lambda: self._call_registry_tool("list_working_files", arguments),
            "calc_materiality": lambda: self._call_registry_tool("calc_materiality", arguments),
            "monetary_unit_sampling": lambda: self._call_registry_tool("monetary_unit_sampling", arguments),
        }
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"未知工具：{name}")
        text = handler()
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": {"tool": name, "elapsed_ms": int((time.perf_counter() - started) * 1000)},
            "isError": False,
        }

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        if method is None or method.startswith("notifications/"):
            return None  # 通知类消息不回响应
        if method == "initialize":
            requested = str(params.get("protocolVersion") or "")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            return _ok(
                request_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": "审计合规知识库 MCP 服务：准则类问题先检索再作答，计算类问题调用确定性工具。",
                },
            )
        if method == "ping":
            return _ok(request_id, {})
        if method == "tools/list":
            return _ok(request_id, {"tools": TOOL_SPECS})
        if method == "tools/call":
            name = str(params.get("name") or "")
            try:
                return _ok(request_id, self.call_tool(name, params.get("arguments") or {}))
            except Exception as exc:  # noqa: BLE001 - 工具异常转 isError，不打断协议
                log(f"tools/call 失败：{name}: {exc}")
                return _ok(
                    request_id,
                    {"content": [{"type": "text", "text": f"工具执行失败：{type(exc).__name__}: {exc}"}], "isError": True},
                )
        if request_id is None:
            return None
        return _error(request_id, -32601, f"未实现的方法：{method}")


def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(stdin=None, stdout=None) -> int:
    """主循环：按行读取 JSON-RPC 报文，逐条响应。"""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = AuditMcpServer()
    log(f"启动 {SERVER_NAME} v{SERVER_VERSION}（stdio）")
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_error(None, -32700, "Parse error"), ensure_ascii=False) + "\n")
            stdout.flush()
            continue
        response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
