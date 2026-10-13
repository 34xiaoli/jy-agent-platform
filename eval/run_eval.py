"""金标集评测：检索命中（Recall@k / MRR）+ 引用合规 + 拒答正确性 + 可选 LLM-as-Judge。

为什么要自建评测而不是"感觉还行"：
- 检索质量必须可量化，否则换切片策略、换嵌入模型、调 top_k 都只能凭感觉；
- 拒答能力是合规场景的核心指标（宁可拒答，不可编造），必须单独度量；
- 评测集是回归资产：提示词、模型、检索参数每次改动都要跑一遍，防止悄悄变差。

用法：
    python eval/run_eval.py                       # mock 后端（离线可跑）
    LLM_PROVIDER=ollama python eval/run_eval.py   # 本地模型
    python eval/run_eval.py --top-k 3 --no-judge
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from scripts.ingest import build_index  # noqa: E402

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.jsonl"
RELEASE_BASELINE = {
    "delivered_hit": 0.8,
    "recall_at_k": 0.6,
    "decision_accuracy": 0.9,
    "citation_compliance": 0.9,
    "judge": 0.7,
}
JUDGE_PROMPT = """请评估以下回答是否忠实于检索依据，只输出 JSON。
问题：{question}
依据：{evidence}
回答：{answer}
输出：{{"score": 0-1 之间的小数, "reason": "一句话理由"}}"""


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def matches(citation: str, expectations: list[str]) -> bool:
    """期望值支持前缀匹配（以 · 结尾表示层级 / 文档级）。"""
    return any(citation == item or citation.startswith(item) for item in expectations)


def reciprocal_rank(citations: list[str], expectations: list[str]) -> float:
    for rank, citation in enumerate(citations, start=1):
        if matches(citation, expectations):
            return 1.0 / rank
    return 0.0


def judge(llm, question: str, evidence: str, answer: str) -> float | None:
    """LLM-as-Judge 忠实度评分：解析失败返回 None，不污染整体统计。"""
    try:
        result = llm.chat(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(question=question, evidence=evidence[:3000], answer=answer[:2000]),
                }
            ],
            temperature=0.0,
            max_tokens=256,
        )
        payload = json.loads(result.text[result.text.find("{") : result.text.rfind("}") + 1])
        return float(payload.get("score"))
    except Exception:  # noqa: BLE001 - 评测环节异常不应中断整轮
        return None


def run_cases(orchestrator: Orchestrator, cases: list[dict[str, Any]], top_k: int, llm, use_judge: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        state, _ = orchestrator.answer(case["question"], role=case.get("role", "assistant"), top_k=top_k)
        delivered = list(state.citations)
        ranking = list(state.retrieval.citations) if (state.retrieval and state.retrieval.citations) else delivered
        expectations = case.get("expect_any") or []
        expect_refuse = bool(case.get("expect_refuse"))
        score = None
        if use_judge and not state.refused:
            context = state.retrieval.context if state.retrieval else ""
            score = judge(llm, case["question"], context, state.answer)
        rows.append(
            {
                "id": case["id"],
                "note": case.get("note", ""),
                "expect": expectations,
                "intent": state.intent,
                # 交付命中：期望引用出现在最终交付给用户的引用里（对用户最有意义的指标）
                "hit": bool(expectations) and any(matches(item, expectations) for item in delivered),
                # 排序命中：期望引用出现在检索排序的 top-k 内（衡量检索与重排本身）
                "rank_hit": bool(expectations) and any(matches(item, expectations) for item in ranking[:top_k]),
                "rr": round(reciprocal_rank(ranking, expectations), 3) if expectations else 0.0,
                "retrieved": ranking[:3],
                "refused": state.refused,
                "decision_ok": state.refused == expect_refuse,
                "verified": state.verified,
                "compliant": state.refused or (state.verified and bool(state.citations)),
                "judge": score,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], top_k: int, settings, index_size: int, summary: dict[str, int], elapsed: float) -> tuple[str, dict[str, float], bool]:
    answered = [row for row in rows if row["expect"]]
    delivered_hit = sum(1 for row in answered if row["hit"]) / len(answered) if answered else 0.0
    recall = sum(1 for row in answered if row["rank_hit"]) / len(answered) if answered else 0.0
    rr_values = [row["rr"] for row in answered]
    mrr = statistics.fmean(rr_values) if rr_values else 0.0
    decision_acc = sum(1 for row in rows if row["decision_ok"]) / len(rows) if rows else 0.0
    compliance = sum(1 for row in rows if row["compliant"]) / len(rows) if rows else 0.0
    judge_scores = [row["judge"] for row in rows if row["judge"] is not None]
    judge_avg = statistics.fmean(judge_scores) if judge_scores else None
    metrics = {
        "delivered_hit": delivered_hit,
        "recall_at_k": recall,
        "mrr": mrr,
        "decision_accuracy": decision_acc,
        "citation_compliance": compliance,
        "judge": judge_avg if judge_avg is not None else -1.0,
    }

    lines = [
        "# 评测报告（自动生成，请勿手工修改）",
        "",
        f"- 推理后端：`{settings.provider}` / `{settings.resolved_chat_model}`；嵌入：`{settings.resolved_embed_model}`",
        f"- 索引规模：{index_size} 块；层级分布 `{summary}`",
        f"- 金标集：{len(rows)} 条（含引用期望 {len(answered)} 条，其余为应拒答用例）；耗时 {elapsed:.2f}s",
        "",
        f"> 说明：mock 后端使用哈希伪向量，`Recall@{top_k}` 只用于验证链路连通性；",
        "> 切换到真实嵌入模型（BGE-M3 / Qwen3-Embedding）后必须重新基线化。",
        "",
        "| 指标 | 数值 | 发布基线 | 含义 |",
        "|---|---|---|---|",
        f"| 交付命中率 | **{delivered_hit:.3f}** | ≥ {RELEASE_BASELINE['delivered_hit']} | 期望引用出现在最终交付引用中 |",
        f"| Recall@{top_k} | **{recall:.3f}** | ≥ {RELEASE_BASELINE['recall_at_k']} | 期望引用出现在检索排序 top-{top_k} |",
        f"| MRR | **{mrr:.3f}** | — | 期望引用平均首位排名倒数 |",
        f"| 决策正确率 | **{decision_acc:.3f}** | ≥ {RELEASE_BASELINE['decision_accuracy']} | 该答的答、该拒的拒 |",
        f"| 引用合规率 | **{compliance:.3f}** | ≥ {RELEASE_BASELINE['citation_compliance']} | 回答带可溯源引用，或按规则拒答 |",
        f"| Judge 忠实度 | {f'{judge_avg:.3f}' if judge_avg is not None else '—'} | ≥ {RELEASE_BASELINE['judge']} | LLM-as-Judge 对依据的忠实程度 |",
        "",
        "## 逐题结果",
        "",
        "| ID | 意图 | 期望引用 | 交付命中 | 排序命中 | RR | 拒答 | 决策 | 校验 | 检索前三 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['intent']} | {'；'.join(row['expect']) or '（应拒答）'} | "
            f"{'✅' if row['hit'] else '—'} | {'✅' if row['rank_hit'] else '—'} | {row['rr']} | "
            f"{'是' if row['refused'] else '否'} | {'✅' if row['decision_ok'] else '❌'} | "
            f"{'✅' if row['verified'] else '❌'} | {'；'.join(row['retrieved'])} |"
        )
    passed = (
        delivered_hit >= RELEASE_BASELINE["delivered_hit"]
        and recall >= RELEASE_BASELINE["recall_at_k"]
        and decision_acc >= RELEASE_BASELINE["decision_accuracy"]
        and compliance >= RELEASE_BASELINE["citation_compliance"]
        and (judge_avg is None or judge_avg >= RELEASE_BASELINE["judge"])
    )
    lines.append("")
    lines.append(f"**结论：{'通过（达到发布基线）' if passed else '未达标（低于发布基线，禁止上线）'}**")
    return "\n".join(lines) + "\n", metrics, passed


def main() -> int:
    parser = argparse.ArgumentParser(description="金标集评测")
    parser.add_argument("--golden", default=str(GOLDEN_SET))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-Judge（省 token）")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "report.md"))
    args = parser.parse_args()

    settings = get_settings()
    llm = build_llm(settings)
    started = time.perf_counter()
    index, _, summary = build_index(settings, verbose=False)
    orchestrator = Orchestrator(llm=llm, retriever=Retriever(index, llm, settings), settings=settings)
    rows = run_cases(orchestrator, load_golden_set(Path(args.golden)), args.top_k, llm, not args.no_judge)
    report, metrics, passed = summarize(rows, args.top_k, settings, len(index), summary, time.perf_counter() - started)

    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"[eval] 报告写入 {args.out}")
    print(
        f"[eval] Recall@{args.top_k}={metrics['recall_at_k']:.3f} MRR={metrics['mrr']:.3f} "
        f"决策正确率={metrics['decision_accuracy']:.3f} 引用合规率={metrics['citation_compliance']:.3f}"
    )
    print("[eval] 结论：" + ("通过" if passed else "未达标"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

