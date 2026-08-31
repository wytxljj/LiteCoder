"""CLI 入口：一次性任务模式与交互式 REPL。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from .agent import Agent
from .config import Config
from .llm import LLMClient, LLMError
from .tools import ToolRegistry


def _print_trace(trace: list[dict], success: bool) -> None:
    if not trace:
        return
    print("\n" + "─" * 60)
    print("执行轨迹（Execution Trace）")
    print("─" * 60)
    for item in trace:
        args = item["args"] if item["args"] is not None else {}
        print(f"Step {item['step']:<2}  {item['tool']}({args})")
    print("─" * 60)
    modified = sum(
        1
        for item in trace
        if item["tool"] in ("write_file", "edit_file")
        and item["args"]
        and "path" in item["args"]
    )
    status = "Success" if success else "Failed"
    print(f"Summary: {len(trace)} tool calls / {modified} file(s) modified / {status}")


def _report_llm_failure(agent: Agent, exc: LLMError) -> None:
    """LLM API 失败时的优雅降级：打印已完成的轨迹与友好错误，避免 traceback 崩溃。"""
    _print_trace(agent.trace, success=False)
    print("\n" + "=" * 60)
    print(f"❌ 模型 API 调用失败，任务中断：{exc}")
    print("=" * 60)
    print("已完成的部分执行轨迹见上；请检查网络或 API 配置后重试。")


def _build_record(
    task: str,
    cfg: Config,
    workspace: Path,
    result: dict | None = None,
    trace: list[dict] | None = None,
    error: Exception | None = None,
) -> dict:
    """构造可回放的完整轨迹记录（成功与失败两种形态）。"""
    record = {
        "task": task,
        "model": cfg.model,
        "workspace": str(workspace),
        "max_steps": cfg.max_steps,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if result is not None:
        record.update(
            {
                "success": result["success"],
                "steps_used": result["steps"],
                "answer": result["answer"],
                "trace": result["trace"],
            }
        )
    else:
        record.update({"success": False, "error": str(error), "trace": trace or []})
    return record


def _write_trace_log(record: dict, path: Path) -> None:
    """把记录写入 JSON 文件（ensure_ascii=False 保证中文可读）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📝 执行轨迹已写入 {path}")


def _numbered_log(path: Path, n: int) -> Path:
    """REPL 模式下给每个任务的日志文件加序号，避免互相覆盖。"""
    path = Path(path)
    return path.with_name(f"{path.stem}-{n}{path.suffix}")


def run_once(task: str, workspace: Path, verbose: bool = True, log_path: Path | None = None) -> int:
    cfg = Config()
    cfg.validate()
    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    agent = Agent(cfg, llm, tools)
    try:
        result = agent.run(task)
    except LLMError as exc:
        if log_path:
            _write_trace_log(_build_record(task, cfg, workspace, trace=agent.trace, error=exc), log_path)
        if verbose:
            _report_llm_failure(agent, exc)
        else:
            print(f"❌ 模型 API 调用失败：{exc}")
        return 1
    finally:
        llm.close()
    if log_path:
        _write_trace_log(_build_record(task, cfg, workspace, result=result), log_path)
    if verbose:
        _print_trace(result["trace"], result["success"])
        print("\n" + "=" * 60)
        print("最终回答")
        print("=" * 60)
    print(result["answer"])
    return 0


def run_repl(workspace: Path, log_path: Path | None = None) -> int:
    cfg = Config()
    cfg.validate()
    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    agent = Agent(cfg, llm, tools)
    print(f"LiteCoder 交互模式（工作区：{workspace}），输入 /quit 退出。")
    counter = 0
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
            try:
                result = agent.run(task)
            except LLMError as exc:
                counter += 1
                if log_path:
                    _write_trace_log(
                        _build_record(task, cfg, workspace, trace=agent.trace, error=exc),
                        _numbered_log(log_path, counter),
                    )
                _report_llm_failure(agent, exc)
                continue
            counter += 1
            if log_path:
                _write_trace_log(
                    _build_record(task, cfg, workspace, result=result),
                    _numbered_log(log_path, counter),
                )
            _print_trace(result["trace"], result["success"])
            print("\n" + result["answer"])
    finally:
        llm.close()
    return 0


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
    parser.add_argument(
        "--log", metavar="PATH", default=None,
        help="将执行轨迹以 JSON 写入指定文件（可回放的完整记录，便于面试展示证据）",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)

    if args.task:
        return run_once(args.task, workspace, verbose=not args.quiet, log_path=args.log)
    return run_repl(workspace, log_path=args.log)


if __name__ == "__main__":
    raise SystemExit(main())
