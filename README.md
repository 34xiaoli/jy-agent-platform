# 审计合规 Agent 平台（jy-agent-platform）

> 面向**审计 / 医疗合规**垂直场景的多智能体平台最小可用实现：
> 分层知识库（L1–L4）+ 混合检索（向量 & BM25 → RRF → 重排）+ 主智能体编排 +
> 子智能体职责拆分 + 确定性工具调用 + 引用校验 + 评测基线 + MCP 工具服务。

**一句话设计哲学：不确定的交给模型，确定的交给代码，权限的交给平台。**
审计场景容不得「差不多对」——引用错一条准则，底稿就是错的。所以这个仓库的重心
不在「让模型答得多聪明」，而在**让每一条结论都能追到条款、每次计算都可复算、
每个角色只看得到该看的**。

---

## 1. 架构

```
┌──────────────────────────────────────────────────────────────┐
│ 用户接入层  项目合伙人 / 项目经理 / 审计助理 + 对话入口        │
├──────────────────────────────────────────────────────────────┤
│ 智能体层    主智能体（意图路由 + 计划 + 汇总 + 引用校验）        │
│            ├─ 风控智能体   风险识别与评级                      │
│            ├─ 审计参谋     准则咨询与依据解释                    │
│            └─ 文书主笔     底稿 / 结论 / 说明 / 报告            │
├──────────────────────────────────────────────────────────────┤
│ 能力服务层  FastAPI：/chat /retrieve /tools /ingest /metrics    │
│            RAG 检索器（混合召回 + RRF + 重排）· 工具注册表        │
│            记忆与上下文裁剪 · Trace 观测 · 鉴权与限流            │
├──────────────────────────────────────────────────────────────┤
│ 知识底座层  L1 法规 / L2 执业准则 / L3 实务经验 / L4 内部数据     │
│            工作区底稿（沙箱）· 工具白名单                       │
└──────────────────────────────────────────────────────────────┘
              推理后端可插拔：mock │ Ollama（本地离线）│ vLLM（生产并发）│ 云 API
              工具接入标准：MCP Server（stdio / JSON-RPC 2.0）
```

## 2. 快速开始

### 2.1 零依赖跑通全链路（推荐先跑这个，不需要 GPU、不需要网络）

```bash
python scripts/smoke_test.py     # 切片→索引→混合检索→重排→编排→引用校验
python eval/run_eval.py          # 12 条金标集评测，输出 eval/report.md
python scripts/mcp_smoke.py      # MCP 协议自检：13 项断言
```

> 核心 RAG / 编排 / 工具 / 评测 / MCP 全部只用**标准库**实现：
> 嵌入默认走 `hash-ngram-256` 伪向量，保证任何机器上都能验证工程正确性。
> 换成真实嵌入模型后必须重新基线化评测指标。

### 2.2 本地离线推理（敏感数据不出机）

```bash
ollama serve
ollama pull qwen3.5:9b && ollama pull bge-m3
# Windows: set LLM_PROVIDER=ollama
export LLM_PROVIDER=ollama          # Linux/macOS
python scripts/ingest.py             # 用真实嵌入重建索引
uvicorn app.main:app --port 8080
```

### 2.3 生产并发推理（vLLM）

```bash
export LLM_PROVIDER=vllm
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export CHAT_MODEL=qwen3.5-9b
export EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B   # 需在同端点暴露 embedding 路由
python scripts/ingest.py
python scripts/bench_llm.py --concurrency 1,4,8,16 --out eval/bench_vllm.md
```

### 2.4 容器化

```bash
docker compose up -d --build                    # 默认 mock 后端，零 GPU 依赖
docker compose --profile local up -d            # + Ollama 本地推理
docker compose --profile prod  up -d            # + vLLM（需 NVIDIA GPU）
docker compose --profile obs   up -d            # + Langfuse 链路观测
curl http://localhost:8080/health
```

### 2.5 MCP 客户端接入

```json
{
  "mcpServers": {
    "audit-compliance": {
      "command": "python",
      "args": ["-m", "mcp.audit_mcp_server"],
      "env": { "LLM_PROVIDER": "mock" }
    }
  }
}
```

暴露 5 个工具：`search_audit_knowledge`（分层混合检索，返回条款级引用）、
`read_workpaper` / `list_workpapers`（沙箱内底稿读写，只读）、
`calc_materiality`（重要性水平）、`monetary_unit_sampling`（货币单元抽样）。

## 3. 目录结构

```
app/
  main.py            FastAPI 能力服务层：/health /chat /retrieve /tools /ingest /metrics /chat/stream
  config.py          集中配置，全部可由环境变量覆盖（后端/检索/治理/观测/路径）
  schemas.py         请求响应模型（pydantic）
  agents/
    orchestrator.py  主智能体：意图路由 → 计划 → 子智能体调度 → 引用校验 → 汇总
    subagents.py     风控 / 审计参谋 / 文书主笔三个子智能体
    prompts.py       提示词模板（含「只依据给定条款作答、必须标注引用」约束）
    state.py         跨轮状态与上下文裁剪
  rag/
    chunker.py       结构化切片：按条款切分，保留层级/权威性/条款号元数据
    store.py         本地混合索引（向量 + BM25 倒排）
    retriever.py     双路召回 → RRF 融合 → 重排 → 带引用的上下文组装
    reranker.py      重排：启发式（零依赖）与 LLM 精排两种实现
  tools/             工具注册表 + 文件类（沙箱只读）+ 计算类（重要性/抽样）
  knowledge/layers.py L1–L4 分层定义与「角色 → 可见层级」权限映射
  context/memory.py  短期会话摘要与上下文预算控制
  obs/tracer.py      Trace/Span 落 JSONL，可选上报 Langfuse
  llm/               推理后端抽象：mock / ollama / openai 兼容（vLLM、云 API）
mcp/audit_mcp_server.py  MCP Server（stdio + JSON-RPC 2.0，零三方依赖）
scripts/
  ingest.py          建索引
  smoke_test.py      端到端冒烟
  mcp_smoke.py       MCP 协议自检（13 项断言）
  bench_llm.py       并发压测（吞吐 / P50 / P95）
  diagnose_coverage.py  拒答阈值标定：打印各题 top-1 词项覆盖率与分离区间
  pdf_check.py       交付前体检：乱码 / 缺字 / 打印页眉页脚 / 版面溢出
  pdf_diff.py        HTML 与 PDF 逐字符清点：确定性回答"有没有字被吞掉"
eval/
  golden_set.jsonl   12 条金标集（含 3 条「应拒答」用例）
  run_eval.py        评测器，输出 eval/report.md
data/knowledge/L1–L4  分层知识库语料（法规 / 准则 / 实务 / 内部制度）
data/workspace/      审计底稿工作区（MCP 文件工具的沙箱根目录）
```

## 4. 六个核心设计决策（可直接作为面试回答）

| # | 决策 | 为什么 |
|---|---|---|
| 1 | **知识库分四层**（L1 法规 / L2 准则 / L3 实务 / L4 内部数据） | 分层依据是**权威性 + 更新频率 + 敏感度**：L1/L2 是「依据」，L3 只能「参考」，L4 仅作事实来源；三层的更新频率决定索引重建节奏，敏感度决定授权策略 |
| 2 | **混合检索（向量 + BM25）+ RRF + 重排** | 向量擅长语义、BM25 擅长精确匹配「CSA 1313 第 2 条」这类编号；RRF 只用排名不用分数尺度，比加权求和更稳；重排负责把对的条款顶到第一位 |
| 3 | **结构化切片 + 条款级引用** | 按条款而非按字数切分，每块携带 `层级/权威性/条款号`，才能生成「可溯源到条款」的引用；相邻条款被合并会直接导致引用张冠李戴 |
| 4 | **权限在检索入口生效，不靠提示词** | `role → 可见层级` 映射在 `retrieve(layers=...)` 处过滤，L4 内部数据对 `intern` 根本不会进入上下文。提示词约束是「劝告」，代码约束才是「保证」 |
| 5 | **确定性的事交给代码** | 重要性水平、货币单元抽样、文件读取全部是工具，模型只负责「决定调不调」和「解释结果」。同一输入永远同一输出，可复算、可审计 |
| 6 | **允许拒答，且阈值是量出来的** | 检索不到依据时明确拒答，而不是用流畅的话术补全。判据是首位候选的**词项覆盖率**：金标集实测"无依据"组最高 0.2121、"有依据"组最低 0.5484，阈值取区间中部 **0.30**。调参用 `scripts/diagnose_coverage.py`，不是拍脑袋定的 |

## 5. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务 / 推理后端 / 索引健康状态 |
| POST | `/chat` | 主智能体问答：结论 + 引用 + 校验结果 + 计划 + `trace_id` + 耗时 |
| POST | `/retrieve` | 仅检索，可对比 `use_hybrid` / `use_rerank` 开关的效果 |
| GET | `/tools` | 已注册工具的 function calling schema |
| POST | `/ingest` | 重建索引 |
| GET | `/metrics` | 基于 trace 的运行指标（请求数、平均耗时、错误 span） |
| GET | `/chat/stream` | SSE 流式：先推 `trace_id` 与引用，再逐行推正文 |

治理能力：可选 `X-API-Key` 鉴权、每 IP 每分钟限流、角色级知识隔离。

示例：

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"审计函证控制在什么情况下必须执行？","role":"assistant","top_k":3}'
```

```json
{
  "answer": "……【CSA 1313 第 2 条】……",
  "verified": true,
  "refused": false,
  "citations": ["【L2·CSA1313·第2条】", "【L2·CSA1313·第1条】"],
  "verification_notes": ["引用均来自 L1/L2 依据层"],
  "trace_id": "3f2a...", "latency_ms": 412
}
```

## 6. 评测

```bash
python eval/run_eval.py           # 输出 eval/report.md
```

| 指标 | 基线 | 含义 |
|---|---|---|
| 交付命中率 | ≥ 0.8 | 期望引用出现在最终交付的引用中 |
| Recall@5 | ≥ 0.6 | 期望引用出现在检索 top-5 |
| MRR | — | 期望引用的平均首位排名倒数 |
| 决策正确率 | ≥ 0.9 | 该答的答、该拒的拒 |
| 引用合规率 | ≥ 0.9 | 回答带可溯源引用，或按规则拒答 |
| Judge 忠实度 | ≥ 0.7 | LLM-as-Judge 对依据的忠实程度 |

三层评测思路：**组件级**（检索召回/重排）→ **端到端**（引用是否真的被用上）→
**业务级**（底稿是否可被复核）。金标集 12 条，含 3 条应拒答用例。

> ✅ 评测器本身就是回归防线：拒答阈值一度被置 0（闸门关闭），
> 评测立刻把 3 条应拒答用例判为失败、决策正确率掉到 0.75 并**拒绝出报告结论**。
> 这正是「先建评测、再调策略」的价值——策略退步会被自动发现，而不是上线后由业务发现。
>
> ⚠️ 当前 mock 后端使用哈希伪向量，`Recall@5` 只证明链路连通。
> 换成 BGE-M3 / Qwen3-Embedding 后**必须重新基线化**，不能拿 mock 的数字对外汇报。

## 7. 性能压测

```bash
python scripts/bench_llm.py --concurrency 1,4,8 --rounds 8
python scripts/bench_llm.py --tag vllm-9b --out eval/bench_vllm.md
```

输出吞吐（req/s）与 P50 / P95 延迟，并自动生成结论口径。
**为什么坚持成对看吞吐与延迟**：推理服务在高并发下会排队，只报吞吐是不诚实的。

- **Ollama** = 本地离线单机（零依赖、模型管理简单、解决**合规**问题）
- **vLLM** = 生产并发（PagedAttention + 连续批处理、解决**并发**问题）
- 两者不冲突：同样使用本地权重，数据不出机，叠加部署即可。

## 8. 已验证 / 待验证（诚实边界）

面试时只讲右边这一列。**没跑过的不要说跑过。**

| 能力 | 当前状态 | 证据 |
|---|---|---|
| 结构化切片 + 条款级引用 | ✅ 已跑通 | 26 块全部带条款级引用 |
| 混合检索 + RRF + 重排 | ✅ 已跑通 | 12 条金标集，交付命中 1.0 / 决策正确率 1.0 |
| 拒答阈值标定 | ✅ 已跑通 | `diagnose_coverage.py` 给出分离区间 (0.2121, 0.5484]，阈值取 0.30 |
| 角色权限隔离（L4 不可见） | ✅ 已跑通 | 评测 Q9–Q12 + MCP 自检双重验证 |
| 应拒答场景 | ✅ 已跑通 | 3 条应拒答用例全部正确拒答 |
| 工具沙箱越权拦截 | ✅ 已跑通 | `../../../etc/passwd` 被 `PermissionError` 拦截 |
| MCP 协议（握手/工具/异常语义） | ✅ 已跑通 | `scripts/mcp_smoke.py` 13 项断言全通过 |
| 离线零依赖可运行 | ✅ 已跑通 | 仅标准库 + 可选 FastAPI |
| Docker / vLLM / 真实嵌入模型 | ⚠️ **代码就绪，本机未执行** | 本机无 Docker / GPU；需自行 `docker compose up` 与 vLLM 压测后补数据 |
| Langfuse 上报链路 | ⚠️ 接口就绪 | 未安装 SDK 时显式降级并回报原因，不静默失败 |

## 9. 面试速答卡（背这三条就够）

**Q：为什么不用微调，非要 RAG？**
审计准则会随监管口径更新，RAG 更新知识只要重建索引，微调要重新训练；
更关键的是**可溯源**——底稿里写「依据 CSA 1313 第 2 条」，必须能点开原文核对。
微调改的是能力与风格，替代不了依据链。

**Q：怎么保证模型不乱说？**
四层：① 检索约束（只把召回条款喂进上下文）② 强制引用（回答必须带条款标记）
③ 后置校验（引用不在候选集内就标记 `verified=false` 并降级）④ 允许拒答
（召回置信度不够就明说「无依据」）。确定性计算全部走工具，模型不参与算数。

**Q：MCP 解决了什么？**
把「工具接入」从一次性集成变成可复用协议：同一套审计工具，
Claude / 飞书 / 任意 MCP 客户端都能直接用，不用为每个客户端重写。
本仓库的 MCP Server 只做只读与纯计算，不暴露写文件与命令执行。

## 10. 边界与免责

- 知识库语料为**演示用精简版**（依据公开准则框架整理），不构成执业依据；
- 默认 mock 后端的评测数字仅证明链路连通，不可作为效果数据对外汇报；
- 本仓库不含任何真实客户数据、底稿或雇主机构资产。

## 11. 相关文档（延伸阅读）

- [《年报审计实质性程序到底稿及报告全流程自动化：AI Agent 落地研究报告》](docs/年报审计AI-Agent落地研究报告.md)
  —— CSA 1312 / 1313 / 1314 拆解、四大与内资所工作流调研、主流审计软件与 AI 产品对比、MVP 路线图
- [《涉密数据 / 图片本地预处理部署方案》](docs/涉密数据本地预处理部署方案.md)
  —— 「全模型离线、数据不出机」的模型与显存调度选型设计


---

技术栈：Python 3.12+ · FastAPI · pydantic v2 · 标准库实现的 RAG 与 MCP ·
可选 Ollama / vLLM / OpenAI 兼容端点 · 可选 Langfuse
