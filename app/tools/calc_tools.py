"""确定性计算工具：重要性水平、货币单元抽样、金额分层抽样。

这些计算交给代码而不是模型，是「不确定性交给模型，确定性交给代码」的落地：
同一个输入永远得到同一个结果，可复算、可审计、可写入底稿。
"""
from __future__ import annotations

import json
import math

from app.tools.registry import Tool


def materiality(benchmark: float, rate: float = 0.05) -> str:
    """计算财务报表整体重要性水平、实际执行重要性与明显微小错报临界值。

    benchmark: 基准金额（如营业收入 / 资产总额 / 税前利润）
    rate: 比例（惯例 0.5%–5%，按基准类型选择）
    """
    if benchmark <= 0:
        raise ValueError("基准金额必须大于 0")
    if not 0 < rate < 1:
        raise ValueError("比例 rate 应在 0 与 1 之间，例如 0.05 表示 5%")
    overall = benchmark * rate
    payload = {
        "benchmark": round(benchmark, 2),
        "rate": rate,
        "overall_materiality": round(overall, 2),
        "performance_materiality": round(overall * 0.75, 2),
        "trivial_threshold": round(overall * 0.05, 2),
        "note": "实际执行重要性通常取整体重要性的 50%–75%；明显微小错报临界值通常取 3%–5%。",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def monetary_unit_sampling(amounts: list[float], sample_size: int) -> str:
    """货币单元抽样（PPS）：项目金额越大，被选中概率越高。

    规则：1) 大于等于抽样间隔的项目必然入选；2) 其余按累计金额等距抽取。
    """
    if not amounts:
        raise ValueError("amounts 不能为空")
    if sample_size <= 0:
        raise ValueError("sample_size 必须为正整数")
    positives = [abs(float(value)) for value in amounts]
    total = sum(positives)
    if total <= 0:
        raise ValueError("金额合计必须大于 0")
    interval = total / sample_size
    high_value = [index for index, amount in enumerate(positives) if amount >= interval]
    targets = [interval / 2 + step * interval for step in range(sample_size)]
    cursor = 0
    cumulative = 0.0
    hits: list[int] = []
    for index, amount in enumerate(positives):
        cumulative += amount
        while cursor < len(targets) and cumulative >= targets[cursor]:
            hits.append(index)
            cursor += 1
    selected = sorted(set(hits) | set(high_value))
    payload = {
        "population_total": round(total, 2),
        "sample_size_target": sample_size,
        "sampling_interval": round(interval, 2),
        "high_value_items": high_value,
        "interval_hits": len(hits),
        "selected_indexes": selected,
        "selected_count": len(selected),
        "selected_total": round(sum(positives[index] for index in selected), 2),
        "coverage": round(sum(positives[index] for index in selected) / total, 4),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def stratified_sample(amounts: list[float], sample_size: int, strata: int = 3) -> str:
    """按金额分层抽样：高、中、低三层按金额占比分配样本量（非统计抽样思路）。"""
    if not amounts or sample_size <= 0:
        raise ValueError("amounts 非空且 sample_size 必须为正整数")
    strata = max(1, min(strata, len(amounts)))
    ordered = sorted(((index, abs(float(value))) for index, value in enumerate(amounts)), key=lambda item: item[1], reverse=True)
    total = sum(amount for _, amount in ordered) or 1.0
    bucket_size = math.ceil(len(ordered) / strata)
    allocation: list[dict[str, object]] = []
    remaining = sample_size
    for layer in range(strata):
        bucket = ordered[layer * bucket_size : (layer + 1) * bucket_size]
        if not bucket:
            continue
        amount_share = sum(amount for _, amount in bucket) / total
        quota = max(1, min(len(bucket), round(sample_size * amount_share)))
        quota = min(quota, remaining)
        remaining -= quota
        allocation.append(
            {
                "stratum": layer + 1,
                "items": len(bucket),
                "amount_share": round(amount_share, 4),
                "quota": quota,
                "selected_indexes": [index for index, _ in bucket[:quota]],
            }
        )
    return json.dumps({"total_items": len(amounts), "allocation": allocation}, ensure_ascii=False, indent=2)


def build_calc_tools() -> list[Tool]:
    return [
        Tool(
            name="calc_materiality",
            description="根据基准金额与比例计算整体重要性水平、实际执行重要性与明显微小错报临界值。",
            parameters={
                "type": "object",
                "properties": {
                    "benchmark": {"type": "number", "description": "基准金额，须大于 0"},
                    "rate": {"type": "number", "description": "比例，默认 0.05 表示 5%"},
                },
                "required": ["benchmark"],
            },
            func=materiality,
            category="audit",
        ),
        Tool(
            name="monetary_unit_sampling",
            description="货币单元抽样（PPS）：按金额大小抽取样本，输出抽样间隔、必选的高价值项目与选中索引。",
            parameters={
                "type": "object",
                "properties": {
                    "amounts": {"type": "array", "items": {"type": "number"}, "description": "总体各项目金额"},
                    "sample_size": {"type": "integer", "description": "目标样本量"},
                },
                "required": ["amounts", "sample_size"],
            },
            func=monetary_unit_sampling,
            category="audit",
        ),
        Tool(
            name="stratified_sample",
            description="按金额分层抽样：将总体分为高/中/低三层并按金额占比分配样本量。",
            parameters={
                "type": "object",
                "properties": {
                    "amounts": {"type": "array", "items": {"type": "number"}, "description": "总体各项目金额"},
                    "sample_size": {"type": "integer", "description": "目标样本量"},
                    "strata": {"type": "integer", "description": "层数，默认 3"},
                },
                "required": ["amounts", "sample_size"],
            },
            func=stratified_sample,
            category="audit",
        ),
    ]
