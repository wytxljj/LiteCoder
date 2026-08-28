"""Agent 核心循环：决策 → 调用工具 → 回填结果 → 再决策，直到终止条件满足。"""
import json

from .config import Config
from .llm import LLMClient
from .tools import ToolRegistry


SYSTEM_PROMPT = """你是一个运行在本地的编程智能体（Coding Agent）。你可以调用工具来读写文件、列出目录、执行命令，从而完成用户的编程任务。

工作方式：
1. 理解用户任务，制定简短计划；
2. 通过调用工具（list_files / read_file / write_file / run_command）逐步执行；
3. 观察每个工具返回的结果，据此决定下一步；
4. 命令失败时阅读报错并尝试修复；
5. 任务完成后，用简洁的中文总结你做了什么、结果如何，不再调用工具。

约束：所有文件读写必须发生在工作区（workspace）内。"""


class Agent:
    """把「LLM 决策」和「本地工具执行」串起来的核心循环。"""

    def __init__(self, cfg: Config, llm: LLMClient, tools: ToolRegistry):
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.trace: list[dict] = []

    def run(self, task: str) -> dict:
        """执行任务，返回 {answer, steps, trace}。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        self.trace = []

        for step in range(1, self.cfg.max_steps + 1):
            message = self.llm.chat(messages, tools=self.tools.schemas)
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # 模型不再调用工具 → 视为任务完成，输出最终答案
                return {
                    "answer": message.get("content") or "(模型未给出最终答案)",
                    "steps": step,
                    "trace": self.trace,
                }

            messages.append(message)  # 带 tool_calls 的 assistant 消息必须回填
            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name", "")
                raw_args = func.get("arguments") or "{}"

                # 解析工具参数（模型输出解析容错：非法 JSON 回喂给模型自纠）
                try:
                    arguments = json.loads(raw_args)
                    if not isinstance(arguments, dict):
                        arguments = {}
                except (json.JSONDecodeError, TypeError):
                    arguments = None

                if arguments is None:
                    result = f"错误：工具参数不是合法 JSON：{raw_args!r}，请重新生成。"
                else:
                    result = self.tools.execute(name, arguments)

                self.trace.append(
                    {"step": step, "tool": name, "args": arguments, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )

        # 达到最大步数仍未结束
        return {
            "answer": f"达到最大步数限制（{self.cfg.max_steps} 步），已停止。",
            "steps": self.cfg.max_steps,
            "trace": self.trace,
        }
