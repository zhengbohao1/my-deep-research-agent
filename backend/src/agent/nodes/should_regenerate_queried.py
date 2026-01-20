from backend.src.agent.states.overallstate import OverallState


def should_regenerate_queried(state: OverallState):
    """路由函数：决定是否需要重新生成问题"""
    # 如果已经收到用户确认，直接进行网络搜索
    if state.get("user_confirmation_received", False):
        return "web_research"
    # 如果需要重新生成问题
    else:
        return "generate_query_node"