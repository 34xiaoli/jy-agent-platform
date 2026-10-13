"""Ollama 推理后端（涉密单机 / 离线场景）。

只用标准库 urllib，不引入任何 SDK，便于在内网无 pip 源的机器上部署。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.llm.base import ChatResult, LLMError, Message

_OLLAMA_MISSING = (
    "无法连接 Ollama。请确认：1) 已安装并启动 ollama serve；"
    "2) 已 ollama pull {model}；3) LLM_BASE_URL 指向正确地址。"
)


class OllamaClient:
    """Ollama 原生 API 客户端（/api/chat、/api/embed）。"""

    name = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.resolved_base_url.rstrip("/")

    # ------------------------------------------------------------ HTTP 基础
    def _request(self, path: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            raise LLMError(f"Ollama 返回 HTTP {exc.code}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(_OLLAMA_MISSING.format(model=self.settings.resolved_chat_model) + f" 原始错误：{exc}") from exc

    # ---------------------------------------------------------------- 生成
    def chat(
        self,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.settings.resolved_chat_model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "options": {
                "temperature": self.settings.temperature if temperature is None else temperature,
                "num_predict": self.settings.max_tokens if max_tokens is None else max_tokens,
            },
        }
        started = time.perf_counter()
        data = self._request("/api/chat", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = str((data.get("message") or {}).get("content", ""))
        if not text:
            raise LLMError("Ollama 返回了空内容，请检查模型是否正常加载（可能是显存不足被卸载）。")
        return ChatResult(
            text=text,
            provider=self.name,
            model=str(data.get("model") or self.settings.resolved_chat_model),
            latency_ms=latency_ms,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    # ---------------------------------------------------------------- 向量
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        data = self._request(
            "/api/embed",
            {"model": self.settings.resolved_embed_model, "input": texts},
        )
        vectors = data.get("embeddings") or []
        if len(vectors) != len(texts):
            raise LLMError(
                f"向量数量与输入不一致（输入 {len(texts)}，返回 {len(vectors)}）。"
                f"请确认已 ollama pull {self.settings.resolved_embed_model}。"
            )
        return [list(map(float, vector)) for vector in vectors]

    # ---------------------------------------------------------------- 健康
    def health(self) -> dict[str, Any]:
        try:
            data = self._request("/api/tags", None, method="GET")
        except LLMError as exc:
            return {"provider": self.name, "ok": False, "detail": str(exc)}
        names = [str(item.get("name", "")) for item in data.get("models", [])]
        chat_model = self.settings.resolved_chat_model
        return {
            "provider": self.name,
            "ok": True,
            "base_url": self.base_url,
            "installed_models": names,
            "chat_model_ready": any(name.startswith(chat_model.split(":")[0]) for name in names),
        }
