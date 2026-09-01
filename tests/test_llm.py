"""LLM 客户端的响应解析容错测试（用 httpx MockTransport，不真发请求）。"""
import httpx
import pytest

from litecoder.config import Config
from litecoder.llm import LLMClient, LLMError


def _client_with_handler(monkeypatch, handler):
    """把 LLMClient 内部的 httpx.Client 换成用指定 handler 的 mock。"""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client  # 先保存真实类，避免 lambda 递归引用
    monkeypatch.setattr(
        httpx, "Client",
        lambda timeout=None: real_client(transport=transport, timeout=timeout),
    )


def test_chat_raises_llm_error_on_bad_json(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(200, text="this is not json")

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    with pytest.raises(LLMError, match="响应结构异常"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_llm_error_on_empty_choices(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(200, json={"choices": []})

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    with pytest.raises(LLMError, match="响应结构异常"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_returns_message_on_success(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    _client_with_handler(monkeypatch, handler)
    llm = LLMClient(Config())
    msg = llm.chat([{"role": "user", "content": "hi"}])
    assert msg["content"] == "hi"
