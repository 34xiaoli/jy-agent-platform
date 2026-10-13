"""审计 / 合规垂类 AI Agent 平台 —— 能力服务层与编排实现。

分层对应关系（五层架构中的第 3–4 层）：
- 第 3 层 智能体层：app/agents/（主智能体路由 + 子智能体）
- 第 4 层 能力服务层：app/tools/、app/rag/、app/llm/、app/main.py
- 知识底座：app/knowledge/ + data/knowledge/L1–L4
"""

__version__ = "0.1.0"
