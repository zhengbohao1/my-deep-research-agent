from langchain_core.runnables import RunnableConfig

from backend.src.agent.format.schema import SummaryOptimization
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.query_pormpt import get_current_date
from backend.src.agent.prompts.summary_optimization_prompt import summary_optimization_instructions
from backend.src.agent.states.overallstate import OverallState


def optimize_summary(state: OverallState, config: RunnableConfig):
    """LangGraph node that optimizes and enhances the research summary.

    Uses quality assessment, fact verification, and relevance analysis to
    create an optimized summary with key insights and actionable items.

    Args:
        state: Current graph state containing all assessment results
        config: Configuration for the runnable

    Returns:
        Dictionary with state update including optimized summary
    """

    # Get original summary
    original_summary = "\n\n---\n\n".join(state["web_research_result"])

    # Format the prompt with all assessment results
    current_date = get_current_date()
    formatted_prompt = summary_optimization_instructions.format(
        current_date=current_date,
        research_topic=state["search_query"],
        original_summary=original_summary,
        quality_assessment=str(state.get("content_quality", {})),
        fact_verification=str(state.get("fact_verification", {})),
        relevance_assessment=str(state.get("relevance_assessment", {}))
    )

    # Initialize DeepSeek
    llm = ModelInstances.answer_model

    result = llm.with_structured_output(SummaryOptimization).invoke(formatted_prompt)

    # Calculate final confidence score
    quality_score = state.get("content_quality", {}).get("quality_score", 0.5)
    fact_confidence = state.get("fact_verification", {}).get("confidence_score", 0.5)
    relevance_score = state.get("relevance_assessment", {}).get("relevance_score", 0.5)
    final_confidence = (quality_score + fact_confidence + relevance_score) / 3

    return {
        "summary_optimization": {
            "optimized_summary": result.optimized_summary,
            "key_insights": result.key_insights,
            "actionable_items": result.actionable_items,
            "confidence_level": result.confidence_level
        },
        "quality_enhanced_summary": result.optimized_summary,
        "final_confidence_score": final_confidence
    }