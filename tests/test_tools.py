"""工具层安全策略与输出截断的单元测试。"""
import pytest

from litecoder.tools import _is_dangerous, _truncate


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -fr /",
        "shutdown",
        "reboot now",
        "halt",
        "poweroff",
        "init 0",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
    ],
)
def test_dangerous_blocked(cmd):
    assert _is_dangerous(cmd) is not None, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest",
        "python -m pytest -v",
        "git diff",
        "rm -rf build/",
        "rm -f temp.txt",
        "rm temp.py",
        "ls -la",
        "echo hello",
    ],
)
def test_safe_allowed(cmd):
    assert _is_dangerous(cmd) is None, cmd


def test_truncate_short_unchanged():
    text = "short output"
    assert _truncate(text) == text


def test_truncate_by_lines_keeps_head_tail():
    # 行数很多、每行较短 → 触发按行截断
    lines = [f"{i:04d}" for i in range(5000)]
    text = "\n".join(lines)
    out = _truncate(text)
    assert out != text
    assert "0000" in out and "0001" in out
    assert "4999" in out and "4998" in out
    assert "省略" in out


def test_truncate_by_chars_when_lines_long():
    # 行数不多但每行超长 → 退化为按字符头尾截断
    text = "\n".join("x" * 300 for _ in range(50))
    out = _truncate(text)
    assert out != text
    assert "省略" in out
    assert len(out) <= 4200
