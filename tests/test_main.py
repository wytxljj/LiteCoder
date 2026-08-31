"""CLI 入口的测试：LLM 失败降级 + 轨迹日志输出。"""
import json

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


class ScriptedLLM:
    """按脚本依次返回预设响应的假客户端。"""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        return self.script.pop(0)

    def close(self):
        pass


def test_run_once_handles_llm_api_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(main_mod, "LLMClient", FailingLLM)
    ret = main_mod.run_once("hello", tmp_path, verbose=True)
    captured = capsys.readouterr()
    assert ret == 1
    assert "API 调用失败" in captured.out
    assert "Traceback" not in captured.out


def test_run_once_writes_trace_log(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    script = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "1", "function": {"name": "list_files", "arguments": '{"path": "."}'}}
            ],
        },
        {"role": "assistant", "content": "done", "tool_calls": []},
    ]
    monkeypatch.setattr(main_mod, "LLMClient", lambda cfg: ScriptedLLM(script))
    log_path = tmp_path / "trace.json"
    ret = main_mod.run_once("hello", tmp_path, verbose=False, log_path=log_path)
    assert ret == 0
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert data["task"] == "hello"
    assert data["success"] is True
    assert data["answer"] == "done"
    assert data["model"] == "deepseek-chat"
    assert len(data["trace"]) == 1
    assert data["trace"][0]["tool"] == "list_files"
    assert data["trace"][0]["args"] == {"path": "."}
