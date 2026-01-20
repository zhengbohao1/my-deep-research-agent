from datetime import datetime


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from backend.src.agent.config.configuration import Configuration
from backend.src.agent.format.schema import SearchQueryList
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.query_pormpt import query_writer_instructions
from backend.src.agent.states.overallstate import OverallState


def generate_query(state: OverallState, config: RunnableConfig,context:str) :
    """LangGraph node that generates search queries based on the User's question.

    Uses LLM to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init DeepSeek
    llm = ModelInstances.query_generator_model
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = datetime.now().strftime("%Y年%m月%d日")
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        number_queries=state["initial_search_query_count"]
    )
    # Generate the search queries
    result = structured_llm.invoke([SystemMessage(content=formatted_prompt),HumanMessage(content=f"上下文消息如下：{context}")])

    print(f"AI回答如下：{result}")
    return Command(update={
        "search_query": result.query,
        "generated_queries": result.query,
        "awaiting_user_confirmation": True,
        "user_confirmation_received": False
    })