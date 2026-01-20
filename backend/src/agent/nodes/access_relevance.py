from langchain_core.runnables import RunnableConfig

from backend.src.agent.format.schema import RelevanceAssessment
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.relevance_assessment_prompt import relevance_assessment_instructions
from backend.src.agent.states.overallstate import OverallState


def assess_relevance(state: OverallState, config: RunnableConfig):
    """LangGraph node that assesses content relevance to the research topic.

    Evaluates how well the content aligns with the research goals, identifies
    covered and missing topics, and provides relevance scoring.

    Args:
        state: Current graph state containing web research results
        config: Configuration for the runnable

    Returns:
        Dictionary with state update including relevance assessment
    """

    # Combine all research content
    combined_content = "\n\n---\n\n".join(state["web_research_result"])

    # Format the prompt
    formatted_prompt = relevance_assessment_instructions.format(
        research_topic=state["search_query"],
        content=combined_content
    )

    # Initialize DeepSeek
    llm = ModelInstances.answer_model

    result = llm.with_structured_output(RelevanceAssessment).invoke(formatted_prompt)

    return {
        "relevance_assessment": {
            "relevance_score": result.relevance_score,
            "key_topics_covered": result.key_topics_covered,
            "missing_topics": result.missing_topics,
            "content_alignment": result.content_alignment
        }
    }