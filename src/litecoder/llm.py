"""OpenAI 兼容接口的裸 HTTP 客户端（不依赖任何 SDK / Agent 框架）。"""
import json
import time
from typing import Any, Callable

import httpx

from .config import Config


class LLMError(Exception):
    """LLM API 调用失败。"""


class LLMClient:
    """通过 httpx 直连 /chat/completions，使用模型原生 Tool Calling（流式累积）。"""

    RETRIABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """发送一次对话请求（流式累积），返回 assistant message（可能带 tool_calls）。

        on_token：可选回调，每当模型流式输出一段文字时调用，用于实时显示最终答案。
        """
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.1,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.cfg.base_url}/chat/completions"

        last_err = ""
        for attempt in range(3):
            try:
                with self._client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", errors="replace")
                        if resp.status_code in self.RETRIABLE_STATUS and attempt < 2:
                            last_err = f"HTTP {resp.status_code}: {body[:200]}"
                            time.sleep(2 ** attempt)
                            continue
                        raise LLMError(f"API 返回 {resp.status_code}: {body[:300]}")
                    return self._parse_stream(resp, on_token)
            except httpx.HTTPError as exc:
                if attempt < 2:
                    last_err = str(exc)
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"网络错误：{exc}") from exc
        raise LLMError(f"重试耗尽：{last_err}")

    @staticmethod
    def _parse_stream(resp, on_token: Callable[[str], None] | None = None) -> dict:
        """解析 SSE 流式响应，累积 content 与 tool_calls，返回完整 message。"""
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        saw_data = False
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            saw_data = True
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"API 响应结构异常（{type(exc).__name__}）：{exc}") from exc
            if delta.get("content"):
                content_parts.append(delta["content"])
                if on_token:
                    on_token(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                entry = tool_calls.setdefault(
                    idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not entry["function"]["name"]:
                    entry["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]
        if not saw_data:
            raise LLMError("API 响应结构异常：不是合法的流式（SSE）响应")
        message = {"role": "assistant", "content": "".join(content_parts) or None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return message

    def close(self) -> None:
        self._client.close()
