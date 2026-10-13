"""端到端冒烟测试：建索引 → 多类问题 → 工具与沙箱校验 → 引用校验兜底。

mock 模式下无需 GPU 与网络即可运行：
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.agents.prompts import REFUSAL_TEXT  # noqa: E402
from app.agents.state import AgentState  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402
from app.obs.tracer import summarize_records  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402
from scripts.ingest import build_index  # noqa: E402

QUESTIONS: list[tuple[str, str]] = [
    ("manager", "分析程序适用哪几个阶段？依据是什么？"),
    ("manager", "货币单元抽样中，账面金额大于等于抽样间隔的项目会被选中吗？"),
    ("manager", "请查看底稿文件，说明本次函证程序的抽样间隔与回函率。"),
    ("manager", "基准 1.2 亿元、比例 0.5%，重要性水平是多少？"),
    ("intern", "处理个人信息应当遵循哪些要求？"),
    ("manager", "请给出该公司收入确认时点的判断依据。"),
]


def check_tools(settings) -> int:
    """工具层校验：正常计算 + 沙箱越权必须被拦截。"""
    registry = build_default_registry(str(settings.workspace_dir))
    failures = 0

    result = registry.call("calc_materiality", {"benchmark": 120000000, "rate": 0.005})
    print(f"[tools] calc_materiality ok={result.ok} 首行={result.output.splitlines()[1].strip() if result.ok else result.error}")
    failures += 0 if result.ok and "600000" in result.output else 1

    sampling = registry.call("monetary_unit_sampling", {"amounts": [800, 300, 120, 60, 20], "sample_size": 2})
    print(f"[tools] monetary_unit_sampling ok={sampling.ok}")
    failures += 0 if sampling.ok and "sampling_interval" in sampling.output else 1

    escape = registry.call("read_text_file", {"path": "../../Windows/win.ini"})
    print(f"[tools] 沙箱越权拦截 ok={not escape.ok} detail={escape.error[:60]}")
    failures += 0 if not escape.ok else 1

    bad_args = registry.call("calc_materiality", {"rate": 0.05})
    print(f"[tools] 参数校验（缺 benchmark）ok={not bad_args.ok}")
    failures += 0 if not bad_args.ok else 1
    return failures


def check_verification_guard(orchestrator: Orchestrator) -> int:
    """引用校验兜底：编造的引用标记必须让回答被降级为拒答。"""
    state = AgentState(question="测试引用校验", role="manager")
    state.answer = "结论：测试。依据：依据【L2·FAKE9999·第 99 条】，见【TOOL·calc_materiality】。"
    orchestrator.verify(state)
    guarded = not state.verified and state.answer.startswith(REFUSAL_TEXT)
    print(f"[guard] 伪造引用被拦截={guarded} 备注={'；'.join(state.verification_notes)[:80]}")
    return 0 if guarded else 1


def main() -> int:
    settings = get_settings()
    print(f"[smoke] provider={settings.provider} chat={settings.resolved_chat_model}")
    llm = build_llm(settings)
    print(f"[smoke] health={llm.health()}")

    failures = check_tools(settings)
    index, _, summary = build_index(settings, verbose=False)
    print(f"[smoke] 索引块数={len(index)} 层级={summary}")

    retriever = Retriever(index, llm, settings)
    orchestrator = Orchestrator(llm=llm, retriever=retriever, settings=settings)
    failures += check_verification_guard(orchestrator)

    passed = 0
    for role, question in QUESTIONS:
        state, _ = orchestrator.answer(question, role=role)
        ok = state.verified or state.refused
        passed += 1 if ok else 0
        # 回归断言：引用标记必须是「层级·文件编号·条款」的短标记，不能夹带正文
        bloated = [citation for citation in state.citations if len(citation) > 32]
        if bloated:
            failures += 1
            print(f"    [断言失败] 引用标记过长，疑似夹带正文：{bloated}")
        first_line = next((line for line in state.answer.strip().splitlines() if line.strip()), "(空)")
        print(f"\n--- [{role}] {question}")
        print(f"    intent={state.intent} verified={state.verified} refused={state.refused} 引用={state.citations}")
        print(f"    摘要：{first_line[:90]}")
        if state.verification_notes:
            print(f"    校验备注：{'；'.join(state.verification_notes)}")

    trace_file = settings.trace_dir / f"{__import__('datetime').date.today().isoformat()}.jsonl"
    print(f"\n[smoke] trace 落盘：{trace_file}，汇总={summarize_records(trace_file)}")
    print(f"[smoke] 问答通过 {passed}/{len(QUESTIONS)}；工具与校验失败数={failures}")
    return 0 if passed == len(QUESTIONS) and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
