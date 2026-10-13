"""OpenAI 兼容后端：同时覆盖 vLLM（生产并发）与云端 API。

vLLM 起服务后即为 OpenAI 兼容协议，只需把 LLM_BASE_URL 指向
http://host:8000/v1，业务代码零改动即可从 Ollama 切到生产级推理。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.llm.base import ChatResult, LLMError, Message


class OpenAICompatClient:
    """OpenAI 兼容协议客户端（/chat/completions、/embeddings、/models）。"""

    name = "openai-compat"

    def __init__(self, settings: Settings | None = None, provider_name: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.resolved_base_url.rstrip("/")
        if provider_name:
            self.name = provider_name

    def _request(self, path: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            hint = "（401/403 请检查 LLM_API_KEY）" if exc.code in (401, 403) else ""
            raise LLMError(f"{self.name} 返回 HTTP {exc.code}{hint}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(
                f"无法连接 {self.base_url}。若使用 vLLM，请确认已启动服务"
                f"（vllm serve <model> --port 8000）。原始错误：{exc}"
            ) from exc

    def chat(
        self,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.settings.resolved_chat_model,
            "messages": [dict(message) for message in messages],
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        data = self._request("/chat/completions", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        choices = data.get("choices") or []
        text = str(((choices[0] if choices else {}).get("message") or {}).get("content", ""))
        if not text:
            raise LLMError(f"{self.name} 返回了空内容，请检查模型名与 max_tokens 设置。")
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            provider=self.name,
            model=str(data.get("model") or self.settings.resolved_chat_model),
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        data = self._request(
            "/embeddings",
            {"model": self.settings.resolved_embed_model, "input": texts},
        )
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [list(map(float, item.get("embedding") or [])) for item in items]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise LLMError(
                f"向量返回异常（输入 {len(texts)}，返回 {len(vectors)}）。"
                f"vLLM 需单独部署嵌入模型（如 Qwen3-Embedding）。"
            )
        return vectors

    def health(self) -> dict[str, Any]:
        try:
            data = self._request("/models", None, method="GET")
        except LLMError as exc:
            return {"provider": self.name, "ok": False, "detail": str(exc)}
        models = [str(item.get("id", "")) for item in data.get("data", [])]
        return {
            "provider": self.name,
            "ok": True,
            "base_url": self.base_url,
            "served_models": models,
            "chat_model_ready": self.settings.resolved_chat_model in models,
        }
