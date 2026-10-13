"""可观测性：Trace/Span 落 JSONL + 可选上报 Langfuse。

默认落 JSONL 的原因：零依赖、可 grep、可直接被评测脚本消费，内网环境也能用；
需要链路可视化时，打开 ENABLE_LANGFUSE 并配置 key（Langfuse 可由 docker compose 一并启动）。
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import Settings, get_settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


@dataclass
class Span:
    name: str
    duration_ms: int = 0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "error": self.error,
        }


class Tracer:
    """一次请求一条 trace：记录每个阶段的耗时、输入输出规模与关键指标。"""

    def __init__(self, settings: Settings | None = None, name: str = "agent.run") -> None:
        self.settings = settings or get_settings()
        self.trace_id = uuid.uuid4().hex[:16]
        self.name = name
        self.started_at = _now_iso()
        self.spans: list[Span] = []
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ 记录
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attributes))
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - 记录失败原因后继续抛出，由上层决定降级
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = int((time.perf_counter() - started) * 1000)
            self.spans.append(span)

    def event(self, name: str, **attributes: Any) -> None:
        self.events.append({"name": name, "at": _now_iso(), **attributes})

    # ------------------------------------------------------------------ 输出
    def finish(self, output: dict[str, Any] | None = None, write: bool = True) -> dict[str, Any]:
        record = {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": _now_iso(),
            "total_ms": sum(span.duration_ms for span in self.spans),
            "spans": [span.as_dict() for span in self.spans],
            "events": self.events,
            "output": output or {},
        }
        if write:
            self._write_jsonl(record)
        if self.settings.enable_langfuse:
            record["langfuse"] = self._push_langfuse(record)
        return record

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        try:
            directory = self.settings.trace_dir
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # 观测失败绝不影响业务
            self.event("trace_write_failed", error=str(exc))

    def _push_langfuse(self, record: dict[str, Any]) -> dict[str, Any]:
        """可选上报：未安装 SDK 时明确降级，而不是静默失败。"""
        try:
            from langfuse import Langfuse  # type: ignore
        except ImportError:
            return {"enabled": True, "ok": False, "detail": "未安装 langfuse SDK：pip install langfuse"}
        try:
            client = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
            trace = client.trace(id=self.trace_id, name=self.name)
            for span in record["spans"]:
                trace.span(
                    name=span["name"],
                    start_time=None,
                    metadata={k: v for k, v in span["attributes"].items() if isinstance(v, (str, int, float, bool))},
                ).end()
            client.flush()
            return {"enabled": True, "ok": True, "host": self.settings.langfuse_host}
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def summarize_records(path) -> dict[str, Any]:
    """读取 JSONL trace 汇总（供 /metrics 与评测使用）。"""
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        return {"traces": 0}
    total = 0
    total_ms = 0
    errors = 0
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            total_ms += int(record.get("total_ms") or 0)
            errors += sum(1 for span in record.get("spans", []) if span.get("status") == "error")
    return {
        "traces": total,
        "avg_total_ms": round(total_ms / total, 1) if total else 0,
        "error_spans": errors,
    }
