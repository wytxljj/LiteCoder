"""CLI 入口的测试：LLM 失败降级 + 轨迹日志 + 会话历史。"""
import json

from litecoder import main as main_mod
from litecoder.llm import LLMError


class FailingLLM:
    """chat 总是抛 LLMError 的假客户端，模拟 API 失败。"""

    def __init__(self, cfg):
        pass

    def chat(self, messages, tools=None, on_token=None):
        raise LLMError("模拟网络失败")

    def close(self):
        pass


class ScriptedLLM:
    """按脚本依次返回预设响应的假客户端。"""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None, on_token=None):
        return self.script.pop(0)

    def close(self):
        pass


def _tool_call(tid, name, args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": tid, "function": {"name": name, "arguments": args}}],
    }


def _final(content):
    return {"role": "assistant", "content": content, "tool_calls": []}


def test_run_once_handles_llm_api_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(main_mod, "LLMClient", FailingLLM)
    ret = main_mod.run_once("hello", tmp_path, verbose=True)
    captured = capsys.readouterr()
    assert ret == 1
    assert "API 调用失败" in captured.out
    assert "Traceback" not in captured.out
    # 失败的会话也会自动存档
    assert list((tmp_path / "sessions").glob("*.json"))


def test_run_once_writes_trace_log(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_SESSIONS_DIR", str(tmp_path / "sessions"))
    script = [
        _tool_call("1", "list_files", '{"path": "."}'),
        _final("done"),
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
    # 记录含 messages（供 --resume 恢复）
    assert data["messages"]
    assert data["messages"][-1]["role"] == "assistant"


def test_sessions_list_and_show(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_SESSIONS_DIR", str(tmp_path / "sessions"))
    script = [
        _tool_call("1", "list_files", "{}"),
        _final("done"),
    ]
    monkeypatch.setattr(main_mod, "LLMClient", lambda cfg: ScriptedLLM(script))
    main_mod.run_once("hello task", tmp_path, verbose=False)

    assert main_mod.main(["--sessions"]) == 0
    out = capsys.readouterr().out
    assert "hello task" in out
    assert "成功" in out

    assert main_mod.main(["--show", "1"]) == 0
    out = capsys.readouterr().out
    assert "hello task" in out
    assert "list_files" in out
    assert "done" in out


def test_resume_continues_from_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_SESSIONS_DIR", str(tmp_path / "sessions"))
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True)
    # 手动放一个「中断」的会话（含断点 messages）
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "resume me"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "function": {"name": "list_files", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "..."},
    ]
    (sessions / "20260901-000000.json").write_text(
        json.dumps({"task": "resume me", "success": False, "trace": [], "messages": msgs}, ensure_ascii=False),
        encoding="utf-8",
    )

    received = {}

    class RecordingLLM:
        def __init__(self, cfg):
            pass

        def chat(self, messages, tools=None, on_token=None):
            received["messages"] = list(messages)  # 快照
            return {"role": "assistant", "content": "resumed", "tool_calls": []}

        def close(self):
            pass

    monkeypatch.setattr(main_mod, "LLMClient", RecordingLLM)
    ret = main_mod.main(["--resume", "1", "-w", str(tmp_path / "ws")])
    assert ret == 0
    # resume 时传给模型的 messages 就是断点内容（而非重新 system+task）
    assert received["messages"] == msgs
    out = capsys.readouterr().out
    assert "resumed" in out
