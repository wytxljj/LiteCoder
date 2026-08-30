"""工具定义与本地执行：JSON Schema 定义 + 执行器，全部自研。"""
import re
import subprocess
from pathlib import Path


MAX_OUTPUT = 4000  # 单个工具返回结果的最大字符数，防止撑爆上下文
HEAD_LINES = 30    # 截断时保留的开头行数
TAIL_LINES = 30    # 截断时保留的结尾行数

# 高危命令拦截：命中即拒绝执行。只拦截语义明确危险、几乎无正当用途的模式，
# 避免误伤正常命令（如相对路径清理 rm -rf build/）。
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-zA-Z]+\s+)*/\s*$", "递归删除根目录（rm -rf /）"),
    (r"\brm\s+(-[a-zA-Z]+\s+)*/\*", "递归删除根目录下所有文件（rm -rf /*）"),
    (r"\brm\s+(-[a-zA-Z]+\s+)*~\s*$", "递归删除家目录（rm -rf ~）"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "关机或重启主机"),
    (r"\binit\s+[06]\b", "切换运行级别导致关机/重启"),
    (r"\bmkfs(\.[a-z0-9]+)?\b", "格式化文件系统"),
    (r"\bdd\b.*\bof=/dev/", "直接覆盖磁盘设备"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb 进程炸弹"),
]


def _is_dangerous(command: str) -> str | None:
    """命中危险模式则返回危险说明，否则返回 None。"""
    for pattern, desc in _DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return desc
    return None


def _truncate(text: str) -> str:
    """超长输出截断：保留头尾关键信息（如 pytest 开头与最终摘要），中间省略。

    优先按行截断（保留头 HEAD_LINES / 尾 TAIL_LINES 行）；单行超长导致
    按字符仍超限时，退化为按字符头尾截断。
    """
    if len(text) <= MAX_OUTPUT:
        return text
    lines = text.splitlines()
    if len(lines) > HEAD_LINES + TAIL_LINES:
        head = "\n".join(lines[:HEAD_LINES])
        tail = "\n".join(lines[-TAIL_LINES:])
        result = (
            f"{head}\n...（中间省略 {len(lines) - HEAD_LINES - TAIL_LINES} 行）...\n{tail}"
        )
        if len(result) <= MAX_OUTPUT:
            return result
    half = MAX_OUTPUT // 2
    return f"{text[:half]}\n...（中间省略，原 {len(text)} 字符）...\n{text[-half:]}"


class ToolError(Exception):
    """工具执行失败（会作为结构化错误返回给模型）。"""


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


LIST_FILES = _schema(
    "list_files",
    "列出工作区（workspace）内某目录下的文件和子目录，返回相对路径。用于先摸清项目结构。",
    {"path": {"type": "string", "description": "要列出的目录，相对工作区；默认 '.'"}},
    [],
)
READ_FILE = _schema(
    "read_file",
    "读取工作区内指定文件的内容（带行号）。用于阅读源代码、测试或报错相关文件。",
    {
        "path": {"type": "string", "description": "文件路径，相对工作区"},
        "offset": {"type": "integer", "description": "起始行号（1 起），默认 1"},
        "limit": {"type": "integer", "description": "最多读取行数，默认 200"},
    },
    ["path"],
)
WRITE_FILE = _schema(
    "write_file",
    "创建新文件或整体重写现有文件。写入前会校验路径必须位于工作区内。",
    {
        "path": {"type": "string", "description": "要写入的文件路径，相对工作区"},
        "content": {"type": "string", "description": "完整的文件内容"},
    },
    ["path", "content"],
)
EDIT_FILE = _schema(
    "edit_file",
    "对工作区内指定文件做局部替换：把 old_text 替换为 new_text。"
    "old_text 必须与文件中某段文本逐字一致（含空格与缩进）。"
    "若匹配不到或匹配到多处会返回错误，请按提示用 read_file 确认后重试。",
    {
        "path": {"type": "string", "description": "要编辑的文件路径，相对工作区"},
        "old_text": {"type": "string", "description": "要被替换的原文，需与文件内容逐字一致"},
        "new_text": {"type": "string", "description": "替换后的新文本"},
    },
    ["path", "old_text", "new_text"],
)
RUN_COMMAND = _schema(
    "run_command",
    "在工作区目录下执行一条 shell 命令，返回 exit_code、stdout、stderr。用于运行测试、脚本、git 等。",
    {
        "command": {"type": "string", "description": "要执行的命令"},
        "timeout": {"type": "integer", "description": "超时秒数，默认 30"},
    },
    ["command"],
)


class ToolRegistry:
    """工具注册与分发：持有 workspace，负责路径安全校验与本地执行。"""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._registry = {
            "list_files": (LIST_FILES, self._list_files),
            "read_file": (READ_FILE, self._read_file),
            "write_file": (WRITE_FILE, self._write_file),
            "edit_file": (EDIT_FILE, self._edit_file),
            "run_command": (RUN_COMMAND, self._run_command),
        }

    @property
    def schemas(self):
        """供 LLM 使用的工具定义列表。"""
        return [schema for schema, _ in self._registry.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """按名称分发执行；任何异常都转成结构化错误文本返回给模型。"""
        if name not in self._registry:
            return f"错误：未知工具 {name!r}，可用工具：{', '.join(self._registry)}"
        _, func = self._registry[name]
        try:
            return func(arguments)
        except ToolError as exc:
            return f"错误：{exc}"
        except Exception as exc:  # 兜底：不让主程序崩溃
            return f"错误：{type(exc).__name__}: {exc}"

    # ---- 路径安全 ----
    def _resolve(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if not (path == self.workspace or str(path).startswith(str(self.workspace) + "/")):
            raise ToolError(f"路径越界：{path_str} 超出工作区范围")
        return path

    # ---- 工具实现 ----
    def _list_files(self, args: dict) -> str:
        path = self._resolve(args.get("path", "."))
        if not path.exists():
            raise ToolError(f"路径不存在：{args.get('path', '.')}")
        if path.is_file():
            return f"[文件] {path.relative_to(self.workspace)}"
        entries = sorted(path.iterdir())
        if not entries:
            return "(空目录)"
        lines = []
        for entry in entries:
            kind = "[目录]" if entry.is_dir() else "[文件]"
            lines.append(f"{kind} {entry.relative_to(self.workspace)}")
        return _truncate("\n".join(lines))

    def _read_file(self, args: dict) -> str:
        path = self._resolve(args["path"])
        if not path.exists():
            raise ToolError(f"文件不存在：{args['path']}")
        if path.is_dir():
            raise ToolError(f"是目录而非文件：{args['path']}")
        offset = max(1, int(args.get("offset", 1)))
        limit = max(1, int(args.get("limit", 200)))
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if offset > len(lines) and lines:
            raise ToolError(f"offset={offset} 超出文件行数（共 {len(lines)} 行）")
        selected = lines[offset - 1 : offset - 1 + limit]
        numbered = [f"{offset + i:>6}  {line}" for i, line in enumerate(selected)]
        result = "\n".join(numbered)
        end = offset - 1 + len(selected)
        if end < len(lines):
            result += f"\n...（共 {len(lines)} 行，已显示 {offset}~{end} 行）"
        return _truncate(result)

    def _write_file(self, args: dict) -> str:
        path = self._resolve(args["path"])
        content = args.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"已写入 {path.relative_to(self.workspace)}（{len(content)} 字符）"

    def _edit_file(self, args: dict) -> str:
        path = self._resolve(args["path"])
        if not path.exists():
            raise ToolError(f"文件不存在：{args['path']}")
        if path.is_dir():
            raise ToolError(f"是目录而非文件：{args['path']}")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        if not old_text:
            raise ToolError("old_text 不能为空")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        total_lines = len(text.splitlines())
        if count == 0:
            raise ToolError(
                f"未找到 old_text（出现 0 次）。文件共 {total_lines} 行。"
                "请用 read_file 重新确认确切文本（注意空格、缩进、标点）。"
            )
        if count > 1:
            raise ToolError(
                f"old_text 出现 {count} 次，存在歧义。请扩大 old_text 以包含更多上下文，"
                "使其在文件中唯一匹配。"
            )
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return (
            f"已替换 {path.relative_to(self.workspace)} 中 1 处匹配"
            f"（{len(old_text)} → {len(new_text)} 字符）"
        )

    def _run_command(self, args: dict) -> str:
        command = args["command"]
        danger = _is_dangerous(command)
        if danger:
            return (
                f"错误：命令被安全策略拦截（{danger}）：{command!r}。"
                "请改用更安全的方式完成同样的目标。"
            )
        timeout = int(args.get("timeout", 30))
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"错误：命令超时（>{timeout}s）：{command}"
        parts = [f"exit_code={proc.returncode}"]
        if proc.stdout:
            parts.append(f"stdout:\n{proc.stdout.rstrip()}")
        if proc.stderr:
            parts.append(f"stderr:\n{proc.stderr.rstrip()}")
        return _truncate("\n".join(parts))
