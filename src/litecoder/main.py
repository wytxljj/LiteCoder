"""CLI 入口：一次性任务模式与交互式 REPL。"""
import argparse
from pathlib import Path

from .agent import Agent
from .config import Config
from .llm import LLMClient
from .tools import ToolRegistry


def _print_trace(trace: list[dict]) -> None:
    if not trace:
        return
    print("\n" + "─" * 60)
    print("执行轨迹（Execution Trace）")
    print("─" * 60)
    for item in trace:
        args = item["args"] if item["args"] is not None else {}
        print(f"Step {item['step']:<2}  {item['tool']}({args})")
    print("─" * 60)
    print(f"共 {len(trace)} 次工具调用")


def run_once(task: str, workspace: Path, verbose: bool = True) -> None:
    cfg = Config()
    cfg.validate()
    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    agent = Agent(cfg, llm, tools)
    try:
        result = agent.run(task)
    finally:
        llm.close()
    if verbose:
        _print_trace(result["trace"])
        print("\n" + "=" * 60)
        print("最终回答")
        print("=" * 60)
    print(result["answer"])


def run_repl(workspace: Path) -> None:
    cfg = Config()
    cfg.validate()
    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    agent = Agent(cfg, llm, tools)
    print(f"LiteCoder 交互模式（工作区：{workspace}），输入 /quit 退出。")
    try:
        while True:
            try:
                task = input("\n>>> ")
            except (EOFError, KeyboardInterrupt):
                break
            task = task.strip()
            if not task:
                continue
            if task in ("/quit", "/exit"):
                break
            result = agent.run(task)
            _print_trace(result["trace"])
            print("\n" + result["answer"])
    finally:
        llm.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="litecoder",
        description="LiteCoder：一个从零实现的轻量级编程智能体（Coding Agent）",
    )
    parser.add_argument("task", nargs="?", help="一次性任务描述（不提供则进入交互模式）")
    parser.add_argument(
        "-w", "--workspace", default=".",
        help="工作区目录（所有文件操作限制在此目录内），默认当前目录",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="只输出最终回答，不打印执行轨迹")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)

    if args.task:
        run_once(args.task, workspace, verbose=not args.quiet)
    else:
        run_repl(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
