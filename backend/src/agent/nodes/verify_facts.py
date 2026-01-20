from langchain_core.runnables import RunnableConfig

from backend.src.agent.format.schema import FactVerification
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.fact_verification_prompt import fact_verification_instructions
from backend.src.agent.prompts.query_pormpt import get_current_date
from backend.src.agent.states.overallstate import OverallState


def verify_facts(state: OverallState, config: RunnableConfig):
    """LangGraph node that verifies facts and claims in the research content.

    Identifies key facts and claims, verifies their accuracy, flags disputed
    information, and provides confidence scores.

    Args:
        state: Current graph state containing web research results
        config: Configuration for the runnable

    Returns:
        Dictionary with state update including fact verification results
    """

    # Combine all research content
    combined_content = "\n\n---\n\n".join(state["web_research_result"])

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = fact_verification_instructions.format(
        current_date=current_date,
        research_topic=state["search_query"],
        content=combined_content
    )

    # Initialize DeepSeek
    llm = ModelInstances.answer_model

    result = llm.with_structured_output(FactVerification).invoke(formatted_prompt)

    return {
        "fact_verification": {
            "verified_facts": result.verified_facts,
            "disputed_claims": result.disputed_claims,
            "verification_sources": result.verification_sources,
            "confidence_score": result.confidence_score
        }
    }