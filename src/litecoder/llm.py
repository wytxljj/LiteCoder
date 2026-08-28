"""OpenAI 兼容接口的裸 HTTP 客户端（不依赖任何 SDK / Agent 框架）。"""
import time
from typing import Any

import httpx

from .config import Config


class LLMError(Exception):
    """LLM API 调用失败。"""


class LLMClient:
    """通过 httpx 直连 /chat/completions，使用模型原生 Tool Calling。"""

    RETRIABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """发送一次对话请求，返回 assistant message（可能带 tool_calls）。"""
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False,
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
                resp = self._client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]
                if resp.status_code in self.RETRIABLE_STATUS and attempt < 2:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                raise LLMError(f"API 返回 {resp.status_code}: {resp.text[:300]}")
            except httpx.HTTPError as exc:
                if attempt < 2:
                    last_err = str(exc)
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"网络错误：{exc}") from exc
        raise LLMError(f"重试耗尽：{last_err}")

    def close(self) -> None:
        self._client.close()
