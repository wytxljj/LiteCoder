"""Agent 核心循环：决策 → 调用工具 → 回填结果 → 再决策，直到终止条件满足。"""
from typing import Callable, Protocol

from .config import Config
from .parsing import parse_arguments
from .tools import ToolRegistry


class ChatModel(Protocol):
    """Agent 依赖的 LLM 抽象（结构化类型）：任何实现 chat / close 的对象均可注入。"""

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        """发送对话请求，返回 assistant message（可能带 tool_calls）。"""
        ...

    def close(self) -> None:
        """释放底层连接。"""
        ...


SYSTEM_PROMPT = """你是一个运行在本地的编程智能体（Coding Agent）。你可以调用工具来读写文件、列出目录、执行命令，从而完成用户的编程任务。

工作方式：
1. 理解用户任务，制定简短计划；
2. 通过调用工具（list_files / read_file / write_file / edit_file / run_command）逐步执行；
3. 观察每个工具返回的结果，据此决定下一步；
4. 命令失败时阅读报错并尝试修复；
5. 任务完成后，用简洁的中文总结你做了什么、结果如何，不再调用工具。

约束：所有文件读写必须发生在工作区（workspace）内。"""


class Agent:
    """把「LLM 决策」和「本地工具执行」串起来的核心循环。"""

    def __init__(
        self,
        cfg: Config,
        llm: ChatModel,
        tools: ToolRegistry,
        on_token: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.on_token = on_token
        self.trace: list[dict] = []
        self.messages: list[dict] = []
        self._streamed = False

    def run(self, task: str, resume_messages: list[dict] | None = None) -> dict:
        """执行任务，返回 {answer, steps, trace, success, streamed, messages}。

        resume_messages 提供时，从该对话状态继续（而非重新从 system+task 开始），
        用于「会话恢复」：中断或达到步数上限后从断点接着跑。
        """
        if resume_messages is not None:
            self.messages = [dict(m) for m in resume_messages]
        else:
            self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]
        self.trace = []
        self._streamed = False

        for step in range(1, self.cfg.max_steps + 1):
            message = self.llm.chat(self.messages, tools=self.tools.schemas, on_token=self._emit_token)
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # 模型不再调用工具 → 视为任务完成，输出最终答案
                self.messages.append(message)  # 最终答案也回填，保证 resume 上下文完整
                return {
                    "answer": message.get("content") or "(模型未给出最终答案)",
                    "steps": step,
                    "trace": self.trace,
                    "success": self._verify_completion(),
                    "streamed": self._streamed,
                    "messages": self.messages,
                }

            self.messages.append(message)  # 带 tool_calls 的 assistant 消息必须回填
            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name", "")
                raw_args = func.get("arguments") or "{}"

                # 解析工具参数（自研容错：剥离多余文本/尾随逗号修复 → 仍失败则回喂模型自纠）
                arguments, parse_error = parse_arguments(raw_args)
                if arguments is None:
                    result = parse_error or "错误：工具参数解析失败。"
                else:
                    result = self.tools.execute(name, arguments)

                trace_item = {"step": step, "tool": name, "args": arguments, "result": result}
                if name == "run_command":
                    trace_item["exit_code"] = self._parse_exit_code(result)
                self.trace.append(trace_item)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )

            # 滑动窗口：裁剪过长的历史，保留 system + 原始任务 + 最近若干轮
            self.messages = self._trim_context(self.messages)

        # 达到最大步数仍未结束
        return {
            "answer": f"达到最大步数限制（{self.cfg.max_steps} 步），已停止。",
            "steps": self.cfg.max_steps,
            "trace": self.trace,
            "success": False,
            "streamed": self._streamed,
            "messages": self.messages,
        }

    def _emit_token(self, token: str) -> None:
        """流式输出回调：转发给外部回调，并标记已流式输出。"""
        if self.on_token:
            self.on_token(token)
            self._streamed = True

    def _trim_context(self, messages: list[dict]) -> list[dict]:
        """滑动窗口：裁剪过长的对话历史，避免撑爆上下文。

        始终保留 system 与首条 user（原始任务）；对中间的 tool 交互按「轮次」整轮
        裁剪（一轮 = 一条 assistant 及其后全部 tool 结果），保证不把 assistant 的
        tool_calls 与其 tool 结果拆散（OpenAI 协议要求二者配对）。
        """
        max_msgs = self.cfg.max_context_messages
        if len(messages) <= max_msgs:
            return messages
        head = messages[:2]
        kept: list[dict] = []
        for round_ in reversed(self._group_rounds(messages[2:])):
            if len(head) + len(kept) + len(round_) > max_msgs:
                break
            kept = round_ + kept
        return head + kept

    @staticmethod
    def _group_rounds(messages: list[dict]) -> list[list[dict]]:
        """按轮次分组：每条 assistant 消息开启一轮，其后的 tool 消息归入同轮。"""
        rounds: list[list[dict]] = []
        current: list[dict] | None = None
        for msg in messages:
            if msg.get("role") == "assistant":
                current = [msg]
                rounds.append(current)
            elif msg.get("role") == "tool":
                if current is not None:
                    current.append(msg)
                else:
                    rounds.append([msg])
        return rounds

    @staticmethod
    def _parse_exit_code(result: str) -> int | None:
        """从 run_command 结果字符串提取 exit_code；超时/拦截等无 exit_code 时返回 None。"""
        first_line = result.split("\n", 1)[0]
        if first_line.startswith("exit_code="):
            try:
                return int(first_line.split("=", 1)[1])
            except ValueError:
                return None
        return None

    def _verify_completion(self) -> bool:
        """独立的任务完成验证：检查最后一次命令执行是否成功。

        模型停止调用工具 ≠ 任务真正完成。若最后一次 run_command 失败（exit_code 非 0，
        或命令超时/被拦截），说明模型在验证尚未通过时就停止了，不能标记为成功。
        若全程没有 run_command（如纯读文件类任务），则任务不依赖命令验证，视为完成。
        """
        for item in reversed(self.trace):
            if item["tool"] == "run_command":
                return item.get("exit_code") == 0
        return True
