"""CLI 入口的优雅降级测试：LLM API 失败时不崩溃。"""
from litecoder import main as main_mod
from litecoder.llm import LLMError


class FailingLLM:
    """chat 总是抛 LLMError 的假客户端，模拟 API 失败。"""

    def __init__(self, cfg):
        pass

    def chat(self, messages, tools=None):
        raise LLMError("模拟网络失败")

    def close(self):
        pass


def test_run_once_handles_llm_api_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(main_mod, "LLMClient", FailingLLM)
    ret = main_mod.run_once("hello", tmp_path, verbose=True)
    captured = capsys.readouterr()
    assert ret == 1
    assert "API 调用失败" in captured.out
    assert "Traceback" not in captured.out  # 不应 traceback 崩溃
