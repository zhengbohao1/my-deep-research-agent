from langchain_core.runnables import RunnableConfig

from backend.src.agent.config.configuration import Configuration
from backend.src.agent.format.schema import ContentQualityAssessment
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.content_quality_prompt import content_quality_instructions
from backend.src.agent.states.overallstate import OverallState


def assess_content_quality(state: OverallState, config: RunnableConfig):
    """LangGraph node that assesses the quality and reliability of research content.

    Evaluates the overall quality of gathered research content, assesses source
    reliability, identifies content gaps, and provides improvement suggestions.

    Args:
        state: Current graph state containing web research results
        config: Configuration for the runnable

    Returns:
        Dictionary with state update including content quality assessment
    """

    # Combine all research content
    combined_content = "\n\n---\n\n".join(state["web_research_result"])

    # Format the prompt
    formatted_prompt = content_quality_instructions.format(
        research_topic=state["search_query"],
        content=combined_content
    )

    # Initialize DeepSeek
    llm = ModelInstances.answer_model

    result = llm.with_structured_output(ContentQualityAssessment).invoke(formatted_prompt)

    print(f"内容质量打分如下：\n{result}")

    return {
        "content_quality": {
            "quality_score": result.quality_score,
            "reliability_assessment": result.reliability_assessment,
            "content_gaps": result.content_gaps,
            "improvement_suggestions": result.improvement_suggestions
        }
    }