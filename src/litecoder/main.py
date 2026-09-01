"""CLI 入口：一次性任务模式、交互式 REPL，以及会话历史（sessions）。"""
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
    modified = _count_modified(trace)
    status = "Success" if success else "Failed"
    print(f"Summary: {len(trace)} tool calls / {modified} file(s) modified / {status}")


def _count_modified(trace: list[dict]) -> int:
    """统计轨迹中修改了多少个文件（write_file / edit_file）。"""
    return sum(
        1
        for item in trace
        if item.get("tool") in ("write_file", "edit_file")
        and item.get("args")
        and "path" in item["args"]
    )


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
    messages: list[dict] | None = None,
) -> dict:
    """构造可回放、可恢复的完整会话记录（成功与失败两种形态）。"""
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
                "messages": result.get("messages", messages or []),
            }
        )
    else:
        record.update(
            {
                "success": False,
                "error": str(error),
                "trace": trace or [],
                "messages": messages or [],
            }
        )
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


# ---- 会话历史（sessions）----


def _session_filename(sessions_dir: Path) -> Path:
    """生成唯一的会话文件名（时间戳 + 冲突序号）。"""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = sessions_dir / f"{base}.json"
    n = 2
    while path.exists():
        path = sessions_dir / f"{base}-{n}.json"
        n += 1
    return path


def _save_session(record: dict, sessions_dir: Path) -> Path | None:
    """自动保存会话记录到 sessions 目录（静默，不打扰正常输出）。"""
    try:
        path = _session_filename(sessions_dir)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except OSError:
        return None


def _load_sessions(sessions_dir: Path) -> list[tuple[str, dict]]:
    """按文件名排序加载所有会话，返回 [(文件名, 记录)]。"""
    if not sessions_dir.exists():
        return []
    out: list[tuple[str, dict]] = []
    for f in sorted(sessions_dir.glob("*.json")):
        try:
            out.append((f.name, json.loads(f.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _resolve_session(sessions_dir: Path, ref: str) -> dict | None:
    """根据序号（1 起）或文件名解析某条会话记录；找不到返回 None。"""
    sessions = _load_sessions(sessions_dir)
    if not sessions:
        return None
    if ref.isdigit():
        idx = int(ref) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx][1]
        return None
    for fname, rec in sessions:
        if ref == fname or ref == fname.removesuffix(".json"):
            return rec
    return None


def _print_sessions(sessions: list[tuple[str, dict]]) -> None:
    """打印历史会话列表（表格）。"""
    if not sessions:
        print("暂无历史会话。跑一次任务后会自动记录。")
        return
    print(f"共 {len(sessions)} 个会话")
    print("─" * 76)
    print(f"{'#':<3}{'时间':<17}{'结果':<7}{'步数':<6}{'改文件':<7}任务")
    print("─" * 76)
    for i, (fname, rec) in enumerate(sessions, 1):
        if rec.get("success"):
            ok = "成功"
        elif "error" in rec:
            ok = "失败"
        else:
            ok = "未完成"
        steps = str(rec.get("steps_used", "-"))
        modified = str(_count_modified(rec.get("trace", [])))
        task = (rec.get("task") or "").replace("\n", " ")[:36]
        ts = fname.removesuffix(".json")
        print(f"{i:<3}{ts:<17}{ok:<7}{steps:<6}{modified:<7}{task}")
    print("─" * 76)
    print("litecoder --show N 查看详情 · litecoder --resume N 继续")


def _show_session(rec: dict) -> None:
    """回放某个会话的完整内容。"""
    print("=" * 60)
    print(f"任务：{rec.get('task', '')}")
    print(f"模型：{rec.get('model', '')}　工作区：{rec.get('workspace', '')}")
    print(f"时间：{rec.get('timestamp', '')}")
    if rec.get("success"):
        status = "成功"
    elif "error" in rec:
        status = f"失败：{rec.get('error', '')}"
    else:
        status = "未完成"
    print(f"结果：{status}")
    trace = rec.get("trace", [])
    if trace:
        print("\n执行轨迹：")
        for item in trace:
            args = item.get("args") or {}
            print(f"  Step {item.get('step')}  {item.get('tool')}({args})")
            result = item.get("result", "")
            if result:
                one_line = result.replace("\n", " ")[:90]
                print(f"         ↳ {one_line}{'…' if len(result) > 90 else ''}")
    answer = rec.get("answer")
    if answer:
        print("\n" + "=" * 60)
        print("最终回答")
        print("=" * 60)
        print(answer)


def _run_with_resume(ref: str, workspace: Path, verbose: bool = True) -> int:
    """从某个会话的断点继续运行。"""
    cfg = Config()
    cfg.validate()
    rec = _resolve_session(cfg.sessions_dir, ref)
    if rec is None:
        print(f"❌ 找不到会话：{ref}（先用 litecoder --sessions 查看）")
        return 1
    messages = rec.get("messages")
    if not messages:
        print("❌ 该会话没有可恢复的上下文（旧格式记录缺 messages 字段）。")
        return 1

    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    on_token = (lambda t: print(t, end="", flush=True)) if verbose else None
    agent = Agent(cfg, llm, tools, on_token=on_token)
    task = rec.get("task") or "(resume)"
    print(f"↻ 从会话 {ref} 继续（任务：{task}）")
    try:
        result = agent.run(task, resume_messages=messages)
    except LLMError as exc:
        _save_session(
            _build_record(task, cfg, workspace, trace=agent.trace, messages=agent.messages, error=exc),
            cfg.sessions_dir,
        )
        _report_llm_failure(agent, exc)
        llm.close()
        return 1
    llm.close()
    _save_session(_build_record(task, cfg, workspace, result=result), cfg.sessions_dir)
    if verbose:
        _print_trace(result["trace"], result["success"])
        if result.get("streamed"):
            print()
        else:
            print("\n" + "=" * 60)
            print("最终回答")
            print("=" * 60)
            print(result["answer"])
    else:
        print(result["answer"])
    return 0


def run_once(task: str, workspace: Path, verbose: bool = True, log_path: Path | None = None) -> int:
    cfg = Config()
    cfg.validate()
    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    on_token = (lambda t: print(t, end="", flush=True)) if verbose else None
    agent = Agent(cfg, llm, tools, on_token=on_token)
    try:
        result = agent.run(task)
    except LLMError as exc:
        _save_session(
            _build_record(task, cfg, workspace, trace=agent.trace, messages=agent.messages, error=exc),
            cfg.sessions_dir,
        )
        if log_path:
            _write_trace_log(
                _build_record(task, cfg, workspace, trace=agent.trace, messages=agent.messages, error=exc),
                log_path,
            )
        if verbose:
            _report_llm_failure(agent, exc)
        else:
            print(f"❌ 模型 API 调用失败：{exc}")
        return 1
    finally:
        llm.close()
    if log_path:
        _write_trace_log(_build_record(task, cfg, workspace, result=result), log_path)
    _save_session(_build_record(task, cfg, workspace, result=result), cfg.sessions_dir)
    if verbose:
        _print_trace(result["trace"], result["success"])
        if result.get("streamed"):
            print()  # 最终回答已流式输出，仅换行收尾
        else:
            print("\n" + "=" * 60)
            print("最终回答")
            print("=" * 60)
            print(result["answer"])
    else:
        print(result["answer"])
    return 0


def run_repl(workspace: Path, log_path: Path | None = None) -> int:
    cfg = Config()
    cfg.validate()
    llm = LLMClient(cfg)
    tools = ToolRegistry(workspace.resolve())
    on_token = (lambda t: print(t, end="", flush=True))
    agent = Agent(cfg, llm, tools, on_token=on_token)
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
                _save_session(
                    _build_record(task, cfg, workspace, trace=agent.trace, messages=agent.messages, error=exc),
                    cfg.sessions_dir,
                )
                if log_path:
                    _write_trace_log(
                        _build_record(task, cfg, workspace, trace=agent.trace, messages=agent.messages, error=exc),
                        _numbered_log(log_path, counter),
                    )
                _report_llm_failure(agent, exc)
                continue
            counter += 1
            _save_session(_build_record(task, cfg, workspace, result=result), cfg.sessions_dir)
            if log_path:
                _write_trace_log(
                    _build_record(task, cfg, workspace, result=result),
                    _numbered_log(log_path, counter),
                )
            _print_trace(result["trace"], result["success"])
            if result.get("streamed"):
                print()  # 已流式输出，仅换行
            else:
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
    parser.add_argument(
        "--sessions", action="store_true",
        help="列出所有历史会话（自动记录于 ~/.litecoder/sessions/）",
    )
    parser.add_argument(
        "--show", metavar="SESSION",
        help="回放某个会话的完整执行轨迹（用序号或文件名）",
    )
    parser.add_argument(
        "--resume", metavar="SESSION",
        help="从某个会话的断点继续运行（用序号或文件名）",
    )
    args = parser.parse_args(argv)

    # 历史会话命令：不需要 workspace / API key，直接处理
    if args.sessions:
        cfg = Config()
        _print_sessions(_load_sessions(cfg.sessions_dir))
        return 0
    if args.show:
        cfg = Config()
        rec = _resolve_session(cfg.sessions_dir, args.show)
        if rec is None:
            print(f"❌ 找不到会话：{args.show}")
            return 1
        _show_session(rec)
        return 0
    if args.resume:
        workspace = Path(args.workspace).resolve()
        if not workspace.exists():
            workspace.mkdir(parents=True, exist_ok=True)
        return _run_with_resume(args.resume, workspace, verbose=not args.quiet)

    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)

    if args.task:
        return run_once(args.task, workspace, verbose=not args.quiet, log_path=args.log)
    return run_repl(workspace, log_path=args.log)


if __name__ == "__main__":
    raise SystemExit(main())
