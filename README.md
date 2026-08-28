# LiteCoder

一个**从零实现**的轻量级编程智能体（Coding Agent）。通过与 LLM 交互，自主读写文件、执行命令，完成交给它的编程任务——类似简化版 Claude Code / Codex / OpenCode。

> 南京大学软件学院 2026 预推免项目考核作品。

## 核心原则

- 不依赖任何 Agent 框架 / SDK（LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen 等）
- 不依赖 API 服务端托管的代码执行 / 文件工具（Code Interpreter、Files API）
- 关键逻辑全部自研：Agent Loop、上下文管理、工具定义与本地执行、模型输出解析、循环终止、错误处理

## 技术栈

- Python 3.11
- 裸 HTTP（httpx）直连 OpenAI 兼容 `/chat/completions`
- 模型原生 Tool Calling（默认 DeepSeek，可切换任意 OpenAI 兼容模型）

## 状态

🚧 开发中，功能与文档逐步完善。

## 运行

（待补充）
