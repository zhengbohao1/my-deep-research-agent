from hello_agents.tools import MemoryTool
from langchain_core.messages import HumanMessage

from backend.src.agent.format.schema import MemoryExtractionOutput
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.memory_prompt import memory_extraction_prompt
from backend.src.agent.states.overallstate import OverallState


def build_memory_extraction_input(state: OverallState) -> str:
    parts = []

    # 1. 只取 HumanMessage 的对话历史（从旧到新）
    human_messages = [
        f"[USER] {msg.content}"
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    ]
    if human_messages:
        parts.append("=== 用户对话历史（仅用户输入） ===\n" + "\n".join(human_messages))

    # 2. 当前轮的用户查询（如果 messages 里没有最新一条，可以单独加）
    if state.get("search_query"):
        parts.append(f"\n=== 本轮用户查询 ===\n{state['search_query']}")

    # 3. 关键子状态（有内容才加）
    if state.get("reflection"):
        ref = state["reflection"]
        parts.append("\n=== Reflection 阶段总结 ===")
        parts.append(f"是否已足够：{ref.get('is_sufficient', '未知')}")
        parts.append(f"知识缺口：{ref.get('knowledge_gap', '无')}")
        if ref.get("follow_up_queries"):
            parts.append(f"后续查询建议：{', '.join(ref['follow_up_queries'])}")

    if state.get("relevance_assessment"):
        rel = state["relevance_assessment"]
        parts.append("\n=== 相关性评估 ===")
        parts.append(f"相关性分数：{rel.get('relevance_score', '未知')}")
        if rel.get("key_topics_covered"):
            parts.append(f"已覆盖主题：{', '.join(rel['key_topics_covered'])}")
        if rel.get("missing_topics"):
            parts.append(f"缺失主题：{', '.join(rel['missing_topics'])}")

    if state.get("summary_optimization"):
        summ = state["summary_optimization"]
        parts.append("\n=== 优化后的总结关键点 ===")
        if summ.get("optimized_summary"):
            parts.append(summ["optimized_summary"][:500])  # 截断防止过长
        if summ.get("key_insights"):
            parts.append("核心洞见：\n" + "\n".join(summ["key_insights"]))
        if summ.get("actionable_items"):
            parts.append("可执行事项：\n" + "\n".join(summ["actionable_items"]))

    if state.get("fact_verification"):
        fv = state["fact_verification"]
        parts.append("\n=== 事实核查结果 ===")
        if fv.get("verified_facts"):
            for fact in fv["verified_facts"][:3]:  # 限制数量
                parts.append(f"- {fact.get('claim', '')} → {fact.get('status', '未知')}")
        if fv.get("disputed_claims"):
            parts.append("有争议的声明：")
            for claim in fv["disputed_claims"][:3]:
                parts.append(f"- {claim.get('claim', '')}")

    if state.get("content_quality"):
        cq = state["content_quality"]
        parts.append("\n=== 内容质量评估 ===")
        parts.append(f"质量分数：{cq.get('quality_score', '未知')}")
        if cq.get("content_gaps"):
            parts.append(f"内容缺失：{', '.join(cq['content_gaps'])}")

    # 4. 其他可能有价值的字段
    if state.get("quality_enhanced_summary"):
        parts.append("\n=== 最终优化总结（简版） ===\n" + state["quality_enhanced_summary"][:300])

    if state.get("verification_report"):
        parts.append("\n=== 验证报告摘要 ===\n" + state["verification_report"][:300])

    # 拼接成一个完整字符串
    full_text = "\n\n".join(parts).strip()
    return full_text if full_text else "本轮无有效对话或状态信息可供分析。"