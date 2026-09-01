"""LLM 客户端的响应解析容错测试（用 httpx MockTransport 模拟 SSE 流式响应）。"""
import json

import httpx
import pytest

from litecoder.config import Config
from litecoder.llm import LLMClient, LLMError


def _sse_body(chunks):
    """把 delta chunk 列表转成 SSE 流式响应体。"""
    lines = []
    for c in chunks:
        lines.append(f"data: {json.dumps(c)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _client_with_handler(monkeypatch, handler):
    """把 LLMClient 内部的 httpx.Client 换成用指定 handler 的 mock。"""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client  # 先保存真实类，避免 lambda 递归引用
    monkeypatch.setattr(
        httpx, "Client",
        lambda timeout=None: real_client(transport=transport, timeout=timeout),
    )


def test_chat_returns_content(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        body = _sse_body([{"choices": [{"delta": {"content": "hello"}}]}])
        return httpx.Response(200, text=body)

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    msg = llm.chat([{"role": "user", "content": "hi"}])
    assert msg["content"] == "hello"


def test_chat_accumulates_tool_calls(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "list_files", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": \".\"}"}}]}}]},
        ]
        return httpx.Response(200, text=_sse_body(chunks))

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    msg = llm.chat([{"role": "user", "content": "hi"}])
    assert msg["tool_calls"] == [{"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"}}]


def test_chat_raises_llm_error_on_non_sse(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(200, text="not an sse stream")

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    with pytest.raises(LLMError, match="响应结构异常"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_llm_error_on_empty_choices(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        body = _sse_body([{"choices": []}])
        return httpx.Response(200, text=body)

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    with pytest.raises(LLMError, match="响应结构异常"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_calls_on_token(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        body = _sse_body([
            {"choices": [{"delta": {"content": "he"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
        ])
        return httpx.Response(200, text=body)

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    tokens = []
    msg = llm.chat([{"role": "user", "content": "hi"}], on_token=tokens.append)
    assert tokens == ["he", "llo"]
    assert msg["content"] == "hello"
