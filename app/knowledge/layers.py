"""L1–L4 分层知识库：分层依据是「权威性 + 更新频率 + 使用方式」。

为什么必须分层（可直接作为面试回答的三点）：
1. 权威性不同 → 引用强度不同：L1/L2 是「依据」，L3 只能「参考」，L4 仅作事实来源；
2. 更新频率不同 → 切片与重建策略不同（L1 低频、L3 高频，索引重建节奏不一样）；
3. 敏感度不同 → 权限隔离不同（L4 内部数据必须按角色授权，不能对所有人可见）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from app.rag.chunker import Chunk


@dataclass(frozen=True)
class LayerSpec:
    code: str
    name: str
    authority: str
    description: str
    update_frequency: str
    citation_role: str


LAYERS: dict[str, LayerSpec] = {
    "L1": LayerSpec(
        code="L1",
        name="法律法规",
        authority="最高",
        description="法律、行政法规、部门规章（如数据安全、个人信息保护相关规定）",
        update_frequency="低频",
        citation_role="依据",
    ),
    "L2": LayerSpec(
        code="L2",
        name="执业准则",
        authority="高",
        description="审计准则、应用指南等职业规范，按准则编号形成稳定 doc_id",
        update_frequency="中频（版本化）",
        citation_role="依据",
    ),
    "L3": LayerSpec(
        code="L3",
        name="实务经验",
        authority="中",
        description="案例、经验总结、内部作业指引，用于解释与扩展，不能单独作为依据",
        update_frequency="高频",
        citation_role="参考",
    ),
    "L4": LayerSpec(
        code="L4",
        name="内部数据",
        authority="事实来源",
        description="底稿、台账、系统导出的业务数据，敏感度最高，需按角色授权",
        update_frequency="高频",
        citation_role="事实",
    ),
}

LAYER_ORDER = ("L1", "L2", "L3", "L4")

# 角色 → 可见层级（权限隔离在检索入口处生效，而不是靠提示词约束）
ROLE_LAYERS: dict[str, tuple[str, ...]] = {
    "intern": ("L1", "L2"),
    "assistant": ("L1", "L2", "L3"),
    "manager": ("L1", "L2", "L3", "L4"),
    "partner": ("L1", "L2", "L3", "L4"),
}


def layer_specs() -> list[dict[str, str]]:
    return [asdict(LAYERS[code]) for code in LAYER_ORDER]


def layer_spec(code: str) -> LayerSpec | None:
    return LAYERS.get(code.upper())


def allowed_layers(role: str) -> list[str]:
    """按角色返回可见层级；未知角色按最小权限处理。"""
    return list(ROLE_LAYERS.get(role.lower(), ("L1", "L2")))


def summarize_index(chunks: Iterable[Chunk]) -> dict[str, int]:
    """统计各层块数，用于 /health 与索引构建日志。"""
    summary = {code: 0 for code in LAYER_ORDER}
    for chunk in chunks:
        code = chunk.layer.upper()
        summary[code] = summary.get(code, 0) + 1
    return summary
