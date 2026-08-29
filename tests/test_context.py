"""上下文滑动窗口裁剪的单元测试。"""
from pathlib import Path

from litecoder.agent import Agent
from litecoder.config import Config
from litecoder.llm import LLMClient
from litecoder.tools import ToolRegistry


def _make_agent(max_msgs: int) -> Agent:
    cfg = Config()
    cfg.max_context_messages = max_msgs
    return Agent(cfg, LLMClient(cfg), ToolRegistry(Path("/tmp")))


def _round(tid: str) -> list[dict]:
    """构造一轮 assistant(tool_calls) + tool 结果。"""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tid, "function": {"name": "run_command", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": tid, "content": "result"},
    ]


def _messages(round_ids: list[str]) -> list[dict]:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    for tid in round_ids:
        messages.extend(_round(tid))
    return messages


def test_no_trim_when_under_limit():
    agent = _make_agent(max_msgs=100)
    messages = _messages(["1", "2", "3"])
    assert agent._trim_context(messages) == messages


def test_trim_keeps_system_and_task():
    agent = _make_agent(max_msgs=4)
    messages = _messages(["1", "2", "3", "4"])
    trimmed = agent._trim_context(messages)
    assert trimmed[0] == {"role": "system", "content": "S"}
    assert trimmed[1] == {"role": "user", "content": "U"}


def test_trim_keeps_latest_rounds():
    agent = _make_agent(max_msgs=6)
    messages = _messages(["1", "2", "3"])
    trimmed = agent._trim_context(messages)
    ids = [m.get("tool_call_id") for m in trimmed if m["role"] == "tool"]
    assert ids == ["2", "3"]


def test_trim_never_orphans_tool_message():
    agent = _make_agent(max_msgs=5)
    messages = _messages(["1", "2", "3", "4"])
    trimmed = agent._trim_context(messages)
    for i, m in enumerate(trimmed):
        if m["role"] == "tool":
            assert i > 0 and trimmed[i - 1]["role"] == "assistant"
