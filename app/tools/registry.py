"""工具注册表：把「确定性的事」交给代码，是 Agent 不乱来的底座。

设计要点：
1. 每个工具带 JSON Schema，可直接转成 function calling 的 tools 字段；
2. 参数校验 + 异常隔离：工具内部报错不让整条链崩溃，而是把错误回灌给模型重试；
3. 沙箱：文件类工具只允许访问显式白名单根目录，防越权读取。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

ToolFunc = Callable[..., Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc
    category: str = "general"
    requires_confirmation: bool = False

    def spec(self) -> dict[str, Any]:
        """OpenAI / vLLM / Ollama 兼容的 function calling 描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "meta": self.meta,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def extend(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [self._tools[name].spec() for name in self.names()]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """执行工具。任何异常都被转成 ok=False 的结果，由编排层决定重试或降级。"""
        arguments = arguments or {}
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, error=f"未注册的工具：{name}（可用：{self.names()}）")

        missing = [key for key in (tool.parameters.get("required") or []) if key not in arguments]
        if missing:
            return ToolResult(name=name, ok=False, error=f"缺少必填参数：{', '.join(missing)}")

        started = time.perf_counter()
        try:
            output = tool.func(**arguments)
        except TypeError as exc:
            return ToolResult(
                name=name,
                ok=False,
                error=f"参数不匹配：{exc}。期望参数：{list((tool.parameters.get('properties') or {}).keys())}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - 工具异常必须隔离，不能中断编排
            return ToolResult(
                name=name,
                ok=False,
                error=f"工具执行失败：{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        return ToolResult(
            name=name,
            ok=True,
            output=output if isinstance(output, str) else str(output),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def build_default_registry(sandbox_root: str | None = None) -> ToolRegistry:
    """组装默认工具集（文件类 + 计算类）。"""
    from app.tools.calc_tools import build_calc_tools
    from app.tools.fs_tools import build_fs_tools

    registry = ToolRegistry()
    registry.extend(build_fs_tools(sandbox_root))
    registry.extend(build_calc_tools())
    return registry
