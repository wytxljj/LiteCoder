"""模型输出解析容错（parsing.parse_arguments）的单元测试。"""
import pytest

from litecoder.parsing import parse_arguments


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"path": "stats.py"}', {"path": "stats.py"}),
        ('{"path": "stats.py", "limit": 10}', {"path": "stats.py", "limit": 10}),
        ("{}", {}),
        ('{"a": {"b": 1}}', {"a": {"b": 1}}),  # 嵌套对象
    ],
)
def test_valid_json(raw, expected):
    args, err = parse_arguments(raw)
    assert err is None
    assert args == expected


def test_code_fence():
    raw = '```json\n{"path": "a.py"}\n```'
    args, err = parse_arguments(raw)
    assert err is None
    assert args == {"path": "a.py"}


def test_surrounding_text():
    raw = '好的，参数如下：{"path": "a.py"}，请继续。'
    args, err = parse_arguments(raw)
    assert err is None
    assert args == {"path": "a.py"}


def test_trailing_text_after_json():
    raw = '{"path": "a.py"} 这是多余的解释文字'
    args, err = parse_arguments(raw)
    assert err is None
    assert args == {"path": "a.py"}


def test_trailing_comma():
    raw = '{"path": "a.py",}'
    args, err = parse_arguments(raw)
    assert err is None
    assert args == {"path": "a.py"}


def test_braces_inside_string():
    # 字符串值里的花括号不能被误判为对象边界
    raw = '{"content": "print({1, 2, 3})"}'
    args, err = parse_arguments(raw)
    assert err is None
    assert args == {"content": "print({1, 2, 3})"}


def test_garbage():
    args, err = parse_arguments("this is not json at all")
    assert args is None
    assert err is not None


def test_empty():
    args, err = parse_arguments("")
    assert args is None
    assert err is not None


def test_json_but_not_object():
    # 是合法 JSON 但不是对象（列表），工具参数必须为对象 → 视为非法
    args, err = parse_arguments("[1, 2, 3]")
    assert args is None
    assert err is not None
