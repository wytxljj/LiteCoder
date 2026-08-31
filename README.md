# LiteCoder

一个**从零实现**的轻量级编程智能体（Coding Agent）。通过与 LLM 交互，自主读写文件、执行命令，完成交给它的编程任务——类似一个简化版的 Claude Code / Codex / OpenCode。

> 南京大学软件学院 2026 预推免项目考核作品。

## 是什么

LiteCoder 让 LLM 不只是「生成代码」，而是真正动手干活：它反复地 **决策 → 调用本地工具 → 观察结果 → 再决策**，直到把任务做完。你可以交给它「修复这个项目的 bug，让测试通过」这类真实任务，它会自己读代码、跑测试、改文件、再验证，形成完整闭环。

## 核心特性

- **完整闭环**：`list_files / read_file / write_file / edit_file / run_command` 五种工具支撑「读代码 → 跑测试 → 修改 → 再测试」。
- **零 Agent 框架**：不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen 等任何 Agent 框架 / SDK。
- **裸 HTTP + 原生 Tool Calling**：用 httpx 直连 OpenAI 兼容的 `/chat/completions`，不套 openai SDK，自研工具调用协议。
- **关键逻辑全部自研**：Agent Loop、上下文滑动窗口、模型输出解析容错、循环终止、错误处理、工具定义与本地执行。
- **安全边界**：workspace 路径隔离（防 `../../` 越界）+ 危险命令拦截（`rm -rf /`、`shutdown` 等）。
- **可观测执行轨迹**：每一步记录工具调用、参数与结果，末尾输出 `Summary`。

## 架构与工作流程

```
用户任务
   │
   ▼
┌──────────────┐   tool_calls   ┌──────────────┐   执行    ┌──────────────┐
│   LLM 决策   │ ─────────────► │  Tool 分发   │ ────────► │   本地工具   │
│  (chat API)  │ ◄───────────── │  (registry)  │ ◄──────── │  (文件/命令) │
└──────────────┘   tool 结果    └──────────────┘   结果    └──────────────┘
   │
   │ 不再调用工具
   ▼
最终答案
```

每轮循环：模型决定调用哪个工具 → 工具在 workspace 内本地执行 → 结果写回对话历史 → 模型据此决定下一步，直到给出最终答案或达到 `MAX_STEPS` 上限。

## 项目结构

```
litecoder/
├── src/litecoder/
│   ├── agent.py      # Agent Loop + 上下文滑动窗口
│   ├── parsing.py    # 模型输出解析容错（分层修复 + 回喂自纠）
│   ├── tools.py      # 5 个工具的 Schema 定义 + 本地执行 + 安全拦截
│   ├── llm.py        # 裸 httpx 直连 /chat/completions + 重试退避
│   ├── config.py     # 环境变量 / .env 配置（自研 .env 解析，零依赖）
│   └── main.py       # CLI：一次性任务模式 + 交互 REPL
├── tests/            # 单元测试
├── pyproject.toml
├── requirements.txt
└── .env.example      # 配置模板（真实凭据不入库）
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 一个 OpenAI 兼容的模型 API（默认 DeepSeek）

### 2. 安装

```bash
git clone https://github.com/wytxljj/LiteCoder.git
cd LiteCoder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # 安装 litecoder 及其依赖
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY
```

`LLM_API_KEY` 必须通过环境变量或未入库的 `.env` 提供，**绝不**写入代码或提交到仓库。

可选配置（均有默认值）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容 API 地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名，可切 GLM / Qwen 等 |
| `LLM_TIMEOUT` | `120` | 单次 API 请求超时（秒） |
| `MAX_STEPS` | `20` | Agent 最大迭代步数（防无限循环） |
| `MAX_CONTEXT_MESSAGES` | `40` | 上下文滑动窗口保留的消息数 |

### 4. 运行

一次性任务模式（推荐演示，确定性高）：

```bash
litecoder "请检查这个项目，修复 bug 并让所有测试通过" -w /path/to/workspace
```

交互 REPL：

```bash
litecoder -w /path/to/workspace
# >>> 输入任务，/quit 退出
```

保存可回放的执行轨迹（JSON 完整记录，含任务、模型、每一步工具调用与结果、最终答案）：

```bash
litecoder "任务" -w /path/to/workspace --log trace.json
```

## 工具说明

| 工具 | 作用 | 设计要点 |
|---|---|---|
| `list_files(path)` | 列出目录结构 | 先定位项目文件，减少盲目读取 |
| `read_file(path, offset, limit)` | 读取文件（带行号） | 支持分页，限制在 workspace 内 |
| `write_file(path, content)` | 创建 / 整体重写文件 | 写入前做路径校验 |
| `edit_file(path, old_text, new_text)` | 局部替换 | 唯一匹配校验，失败时返回结构化错误引导重试 |
| `run_command(command, timeout)` | 执行命令 | 固定 cwd=workspace，返回 exit_code/stdout/stderr，超长输出头尾截断 |

## 关键设计决策

- **为什么用裸 httpx 而非 openai SDK？** 一是彻底避开「OpenAI Agents SDK」的字面歧义；二是直接掌握 tool calling 协议本身，能证明理解协议细节。
- **为什么同时有 write_file 和 edit_file？** 生成新文件用 write_file；局部修改用 edit_file，比整文件重写更省 token、更安全。
- **模型输出畸形 JSON 怎么办？** `parsing.py` 分层修复（剥代码块 → 提取对象 → 修尾随逗号 → raw_decode），全部失败则把错误回喂模型自纠重试。
- **上下文越来越长怎么办？** 滑动窗口按「轮次」整轮裁剪旧历史，始终保留 system + 原始任务，且不拆散 assistant 的 tool_calls 与其 tool 结果的配对；工具输出超长则头尾截断。
- **如何防止越界 / 危险操作？** 所有文件路径 resolve 后校验仍位于 workspace 内；危险命令（`rm -rf /`、`shutdown`、`mkfs`、`dd of=/dev/`、fork bomb 等）直接拦截。

## 测试

```bash
pip install -e ".[dev]"   # 或手动 pip install pytest
pytest
```

---

本项目为个人独立完成的考核作品，关键逻辑全部自研。
