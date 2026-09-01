"""Agent 核心循环的单元测试（用 mock LLM，不真调 API）。"""
from litecoder.agent import Agent
from litecoder.config import Config
from litecoder.tools import ToolRegistry


class MockLLM:
    """按脚本依次返回预设响应的 mock LLM，并记录每次 chat 的 messages。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return self.script.pop(0)

    def close(self):
        pass


def _assistant_with_tool_call(tid, name, args):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": tid, "function": {"name": name, "arguments": args}}]}


def _final(content):
    return {"role": "assistant", "content": content, "tool_calls": []}


def test_agent_calls_tool_then_finishes(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n")
    cfg = Config()
    llm = MockLLM([
        _assistant_with_tool_call("1", "read_file", '{"path": "a.py"}'),
        _final("done"),
    ])
    agent = Agent(cfg, llm, ToolRegistry(tmp_path))
    result = agent.run("read a.py")
    assert result["success"] is True
    assert result["answer"] == "done"
    assert [t["tool"] for t in result["trace"]] == ["read_file"]


def test_agent_stops_at_max_steps(tmp_path):
    cfg = Config()
    cfg.max_steps = 3
    llm = MockLLM([
        _assistant_with_tool_call("1", "list_files", "{}"),
        _assistant_with_tool_call("2", "list_files", "{}"),
        _assistant_with_tool_call("3", "list_files", "{}"),
    ])
    agent = Agent(cfg, llm, ToolRegistry(tmp_path))
    result = agent.run("loop forever")
    assert result["success"] is False
    assert result["steps"] == 3
    assert "最大步数" in result["answer"]
    assert len(llm.calls) == 3  # 恰好调用 3 次后停止


def test_agent_malformed_json_recovers(tmp_path):
    cfg = Config()
    llm = MockLLM([
        _assistant_with_tool_call("1", "list_files", "not-json"),        # 畸形 JSON
        _assistant_with_tool_call("2", "list_files", '{"path": "."}'),   # 模型自纠
        _final("recovered"),
    ])
    agent = Agent(cfg, llm, ToolRegistry(tmp_path))
    result = agent.run("list")
    assert result["success"] is True
    assert result["trace"][0]["args"] is None           # 第一次解析失败
    assert result["trace"][1]["args"] == {"path": "."}  # 第二次成功


def test_agent_backfills_tool_messages(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    cfg = Config()
    llm = MockLLM([
        _assistant_with_tool_call("1", "read_file", '{"path": "a.py"}'),
        _final("done"),
    ])
    agent = Agent(cfg, llm, ToolRegistry(tmp_path))
    agent.run("read")
    # 第二次 chat 的 messages 应包含完整的 assistant(tool_calls)+tool 配对
    msgs = llm.calls[1]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool"]
    assert msgs[3]["tool_call_id"] == "1"
    assert "x = 1" in msgs[3]["content"]  # 工具结果写回上下文


def test_agent_marks_failure_when_last_command_fails(tmp_path):
    # 模型在最后一次命令失败（exit_code 非 0）时停止，不应标记为成功
    cfg = Config()
    llm = MockLLM([
        _assistant_with_tool_call("1", "run_command", '{"command": "false"}'),
        _final("done"),
    ])
    agent = Agent(cfg, llm, ToolRegistry(tmp_path))
    result = agent.run("run a failing command")
    assert result["success"] is False
    assert result["trace"][0]["exit_code"] == 1  # false 命令返回 1


def test_agent_success_when_last_command_passes(tmp_path):
    # 最后一次命令 exit_code 0 → 成功
    cfg = Config()
    llm = MockLLM([
        _assistant_with_tool_call("1", "run_command", '{"command": "true"}'),
        _final("done"),
    ])
    agent = Agent(cfg, llm, ToolRegistry(tmp_path))
    result = agent.run("run a passing command")
    assert result["success"] is True
    assert result["trace"][0]["exit_code"] == 0
