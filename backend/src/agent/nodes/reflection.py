from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from backend.src.agent.config.configuration import Configuration
from backend.src.agent.format.schema import Reflection
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.query_pormpt import get_current_date
from backend.src.agent.prompts.reflection_prompt import reflection_instructions
from backend.src.agent.states.overallstate import OverallState
from backend.src.agent.states.sub_states.reflectionstate import ReflectionState


def reflection(state: OverallState, config: RunnableConfig) :
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    # 安全获取 reflection（如果不存在，返回默认值）
    reflection = state.get("reflection", {})  # 如果没有，就返回空 dict

    count=reflection.get("research_loop_count",0)+1

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=state["search_query"],
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ModelInstances.answer_model
    result = llm.with_structured_output(Reflection).invoke([SystemMessage(content=formatted_prompt)])

    reflection_result=ReflectionState(is_sufficient=result.is_sufficient,knowledge_gap=result.knowledge_gap,follow_up_queries=result.follow_up_queries,
                                      research_loop_count=count,number_of_ran_queries=len(state["search_query"]))

    print(f"reflection结果如下：\n{reflection_result}")

    return Command(update={"reflection":reflection_result,"search_query":[result.follow_up_queries]})

def evaluate_research(state:OverallState,config:RunnableConfig):
    """LangGraph routing function that determines the next step in the research flow.

        Controls the research loop by deciding whether to continue gathering information
        or to proceed to quality enhancement based on the configured maximum number of research loops.

        Args:
            state: Current graph state containing the research loop count
            config: Configuration for the runnable, including max_research_loops setting

        Returns:
            String literal indicating the next node to visit ("web_research" or "assess_content_quality")
        """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["reflection"]["is_sufficient"] or state["reflection"]["research_loop_count"] >= max_research_loops:
        return "assess_content_quality"
    else:
        return "web_research"