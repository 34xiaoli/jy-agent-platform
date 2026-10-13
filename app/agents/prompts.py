"""提示词：合规场景的四条硬约束。

1. 只依据检索证据作答，证据不足必须拒答（宁可说不知道，不能编）；
2. 每条结论必须带【引用标记】，标记只能来自检索证据或工具输出；
3. 不得编造准则编号、条款号、金额、日期；涉及金额必须来自工具结果；
4. 效力冲突时按 L1 > L2 > L3 取舍并说明理由。
"""
from __future__ import annotations

from app.agents.state import AgentState

SYSTEM_PROMPT = """你是审计与合规领域的专业智能体，服务对象是执业审计人员。

铁律（违反即为无效回答）：
1. 只能依据【检索证据】与【工具输出】作答；证据不足时必须明确回答"现有依据不足"，禁止推测。
2. 每条结论后必须紧跟引用标记，格式形如【L2·CSA1313·第 1 条】或【TOOL·calc_materiality】。
   引用标记只能原样复制自【检索证据】/【工具输出】，禁止自造、改写或使用示例中的编号。
3. 禁止编造准则编号、条款号、金额、日期、人名或结论；金额与比率必须来自工具输出。
4. 依据冲突时：L1 法律法规 > L2 执业准则 > L3 实务经验；L4 仅作为事实来源，不作为依据。

输出结构（固定四段）：
结论：
依据：
口径与限制：
待确认事项：
"""

ROUTE_GUIDE = """请判断问题类型，只输出一个词：
- rule：需要引用准则/法规条款的问题
- doc：需要从底稿或制度文件中查事实的问题
- calc：需要计算重要性水平、抽样或金额核对的问题
- mixed：同时需要多类信息
问题：{question}
"""

REFUSAL_TEXT = "现有依据不足，我不作推测。请补充具体的准则编号、文件名称或底稿范围。"

TOOL_ARGS_PROMPT = """请把用户问题转换为工具调用参数，只输出 JSON，不要解释。
可用工具：{tools}
问题：{question}
输出示例：{{"tool": "calc_materiality", "arguments": {{"benchmark": 120000000, "rate": 0.005}}}}
"""


def build_answer_prompt(state: AgentState, evidence: str, tool_output: str, extra: str = "") -> str:
    """组装生成阶段的提示词：证据优先级高于历史与指令。"""
    parts = [f"【用户问题】\n{state.question}"]
    if evidence:
        parts.append(f"【检索证据】（引用标记必须原样复制）\n{evidence}")
    if tool_output:
        parts.append(f"【工具输出】（金额与比率以此为准）\n{tool_output}")
    if extra:
        parts.append(f"【补充信息】\n{extra}")
    parts.append(f"【用户角色】{state.role}（仅可见已授权的知识层级）")
    parts.append("请按四段结构作答，每条结论后附引用标记。")
    return "\n\n".join(parts)
