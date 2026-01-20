from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.src.agent.states.overallstate import OverallState


def wait_for_user_confirmation(state: OverallState, config: RunnableConfig):
    """LangGraph node that waits for user confirmation of generated queries.

    This node simply outputs the generated queries and waits.
    The workflow will be paused here until user provides confirmation.
    """
    from langchain_core.messages import AIMessage

    # 生成一个包含查询的消息给用户确认
    queries = state.get("generated_queries", state.get("search_query", []))
    confirmation_message = f"我为您生成了以下搜索查询：\n\n" + "\n".join(
        [f"{i + 1}. {q}" for i, q in enumerate(queries)]) + "\n\n请确认是否继续使用这些查询进行搜索，或者您可以修改它们。"

    return {
        "messages": [AIMessage(content=confirmation_message),HumanMessage(content="[查询已确认]")],
        "awaiting_user_confirmation": False,
        "user_confirmation_received": True
    }