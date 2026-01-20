from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from backend.src.agent.states.overallstate import OverallState


def generate_verification_report(state: OverallState, config: RunnableConfig):
    """LangGraph node that generates a comprehensive verification report.

    Creates a detailed report summarizing all quality assessments, fact
    verifications, and enhancement recommendations.

    Args:
        state: Current graph state containing all assessment results
        config: Configuration for the runnable

    Returns:
        Dictionary with state update including verification report
    """
    # Generate comprehensive verification report
    quality_data = state.get("content_quality", {})
    fact_data = state.get("fact_verification", {})
    relevance_data = state.get("relevance_assessment", {})
    optimization_data = state.get("summary_optimization", {})

    report = f"""# 研究质量验证报告

## 内容质量评估
- 质量评分: {quality_data.get('quality_score', 'N/A')}/1.0
- 可靠性评估: {quality_data.get('reliability_assessment', 'N/A')}
- 内容空白: {', '.join(quality_data.get('content_gaps', []))}
- 改进建议: {', '.join(quality_data.get('improvement_suggestions', []))}

## 事实验证结果
- 验证置信度: {fact_data.get('confidence_score', 'N/A')}/1.0
- 已验证事实数量: {len(fact_data.get('verified_facts', []))}
- 争议声明数量: {len(fact_data.get('disputed_claims', []))}
- 验证来源: {', '.join(fact_data.get('verification_sources', []))}

## 相关性评估
- 相关性评分: {relevance_data.get('relevance_score', 'N/A')}/1.0
- 已覆盖关键主题: {', '.join(relevance_data.get('key_topics_covered', []))}
- 缺失主题: {', '.join(relevance_data.get('missing_topics', []))}
- 内容一致性: {relevance_data.get('content_alignment', 'N/A')}

## 摘要优化结果
- 置信度等级: {optimization_data.get('confidence_level', 'N/A')}
- 关键洞察数量: {len(optimization_data.get('key_insights', []))}
- 可行建议数量: {len(optimization_data.get('actionable_items', []))}

## 综合评估
- 最终置信度评分: {state.get('final_confidence_score', 'N/A'):.3f}/1.0
"""

    return {
        "verification_report": report
    }


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the enhanced research summary.

    Creates the final output using the quality-enhanced summary, verification report,
    and properly formatted sources with citations.

    Args:
        state: Current graph state containing the enhanced summary and all assessment results

    Returns:
        Dictionary with state update, including the final enhanced message with sources
    """
    # Use the optimized summary if available, otherwise fall back to original
    final_summary = state.get("quality_enhanced_summary") or "\n---\n\n".join(state["web_research_result"])
    verification_report = state.get("verification_report", "")

    # Combine the enhanced summary with verification report
    enhanced_content = f"""{final_summary}
---

{verification_report}"""

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in enhanced_content:
            enhanced_content = enhanced_content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    # Add quality metrics to the final message
    quality_metrics = f"\n\n## 研究质量指标\n"
    quality_metrics += f"- 最终置信度: {state.get('final_confidence_score', 0):.3f}/1.0\n"
    quality_metrics += f"- 内容质量评分: {state.get('content_quality', {}).get('quality_score', 'N/A')}/1.0\n"
    quality_metrics += f"- 事实验证置信度: {state.get('fact_verification', {}).get('confidence_score', 'N/A')}/1.0\n"
    quality_metrics += f"- 相关性评分: {state.get('relevance_assessment', {}).get('relevance_score', 'N/A')}/1.0\n"

    final_content = enhanced_content + quality_metrics

    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": unique_sources,
    }