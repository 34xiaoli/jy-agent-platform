"""MCP 客户端自检脚本：真正以子进程方式拉起 MCP Server，跑完整协议握手与工具调用。

为什么要有这个脚本
------------------
「MCP Server 写好了」不等于「MCP Server 能用」。协议实现最容易错在
握手版本协商、通知不回响应、异常要转 isError 而非断连这几处。
本脚本用标准库 subprocess + JSON-RPC 客户端把这几条逐条验掉，
输出可直接截图作为「已跑通」的面试证据。

用法：
    python scripts/mcp_smoke.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = "2025-06-18"

PASS, FAIL = "\u2713", "\u2717"
failures: list[str] = []


class McpStdioClient:
    """最小 MCP stdio 客户端：一个请求一个响应，按 id 配对。"""

    def __init__(self) -> None:
        env = dict(os.environ)
        env.setdefault("LLM_PROVIDER", "mock")
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcp.audit_mcp_server"],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Server 提前退出（无响应）")
        return json.loads(line)

    def notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = PASS if condition else FAIL
    print(f"  {mark} {label}" + (f" —— {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def text_of(response: dict) -> str:
    parts = (response.get("result") or {}).get("content") or []
    return "\n".join(str(part.get("text", "")) for part in parts)


def main() -> int:
    print("MCP Server 自检（stdio / JSON-RPC 2.0）")
    client = McpStdioClient()
    try:
        init = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "mcp-smoke", "version": "0.1.0"},
            },
        )
        result = init.get("result") or {}
        check("initialize 返回协议版本", result.get("protocolVersion") == PROTOCOL, str(result.get("protocolVersion")))
        check("serverInfo 正确", (result.get("serverInfo") or {}).get("name") == "jy-audit-compliance")
        check("声明 tools 能力", "tools" in (result.get("capabilities") or {}))

        client.notify("notifications/initialized")
        check("ping 正常", "result" in client.request("ping"))

        tools = (client.request("tools/list").get("result") or {}).get("tools") or []
        names = [tool.get("name") for tool in tools]
        check("tools/list 返回 5 个工具", len(tools) == 5, "、".join(str(n) for n in names))
        check("每个工具都有 inputSchema", all(tool.get("inputSchema") for tool in tools))

        search = client.request(
            "tools/call",
            {"name": "search_audit_knowledge", "arguments": {"query": "函证控制在什么情况下必须执行", "role": "assistant", "top_k": 3}},
        )
        body = text_of(search)
        check("检索工具返回条款级引用", "【L2·" in body, body.splitlines()[0][:60] if body else "无输出")

        denied = client.request(
            "tools/call",
            {"name": "search_audit_knowledge", "arguments": {"query": "内部底稿台账汇总", "role": "intern", "top_k": 5}},
        )
        check("角色权限隔离：intern 看不到 L4", "L4·" not in text_of(denied), "L4 已在检索入口过滤")

        materiality = client.request(
            "tools/call", {"name": "calc_materiality", "arguments": {"benchmark": 100000000, "rate": 0.05}}
        )
        check("重要性水平计算正确", "5000000" in text_of(materiality), "100000000 × 5% = 5,000,000")

        pps = client.request(
            "tools/call",
            {"name": "monetary_unit_sampling", "arguments": {"amounts": [5000000, 120000, 88000, 45000], "sample_size": 2}},
        )
        check("货币单元抽样返回选中索引", "selected_indexes" in text_of(pps))

        files = client.request("tools/call", {"name": "list_workpapers", "arguments": {"pattern": "*.md"}})
        check("工作目录可列出文件", "底稿" in text_of(files))

        escaped = client.request(
            "tools/call", {"name": "read_workpaper", "arguments": {"path": "../../../etc/passwd"}}
        )
        check("沙箱越权被拒绝", bool((escaped.get("result") or {}).get("isError")), "路径逃逸已拦截")

        unknown = client.request("tools/call", {"name": "not_exist", "arguments": {}})
        check("未知工具返回 isError 而非断连", bool((unknown.get("result") or {}).get("isError")))

        missing = client.request("nope/method")
        check("未实现方法返回 JSON-RPC 错误码", (missing.get("error") or {}).get("code") == -32601)
    finally:
        client.close()

    print()
    if failures:
        print(f"自检未通过，失败 {len(failures)} 项：{'、'.join(failures)}")
        return 1
    print("自检全部通过：协议握手、工具发现、权限隔离、沙箱约束、异常语义均正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
