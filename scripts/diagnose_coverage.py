"""诊断：打印金标集每道题的 top-1 词项覆盖率与召回数，用于标定拒答阈值。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.layers import allowed_layers  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.rag.store import LocalHybridIndex  # noqa: E402

settings = get_settings()
llm = build_llm(settings)
index = LocalHybridIndex.load(settings.index_file)
retriever = Retriever(index, llm, settings)

rows = []
for line in (REPO_ROOT / "eval" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    case = json.loads(line)
    result = retriever.retrieve(case["question"], top_k=5, layers=allowed_layers(case["role"]))
    coverage = float(result.items[0].detail.get("coverage", 0.0)) if result.items else 0.0
    rows.append((case["id"], case["expect_refuse"], round(coverage, 4), len(result.items), case["question"]))

print(f"{'ID':<5}{'应拒答':<8}{'top1覆盖率':<12}{'召回数':<8}问题")
for case_id, refuse, coverage, count, question in rows:
    print(f"{case_id:<5}{str(refuse):<8}{coverage:<12}{count:<8}{question}")

should_refuse = [coverage for _, refuse, coverage, _, _ in rows if refuse]
should_answer = [coverage for _, refuse, coverage, _, _ in rows if not refuse]
print()
print(f"应拒答组覆盖率：{should_refuse}")
print(f"应回答组覆盖率：{should_answer}")
if should_refuse and should_answer:
    print(f"\n分离区间：({max(should_refuse)} , {min(should_answer)}]  ← 阈值取区间中点 {(max(should_refuse) + min(should_answer)) / 2:.3f}")
