import operator
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from backend.src.agent.states.sub_states.contentqualitystate import ContentQualityState
from backend.src.agent.states.sub_states.factverificationstate import FactVerificationState
from backend.src.agent.states.sub_states.reflectionstate import ReflectionState
from backend.src.agent.states.sub_states.relevancestate import RelevanceState
from backend.src.agent.states.sub_states.summaryoptimizationstate import SummaryOptimizationState
from backend.src.agent.states.sub_states.translationstate import TranslationState


class OverallState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    search_query: Annotated[List[str], operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    # research_topic:str#新增，用户关注的查询主题。
    # human-in-the-loop相关状态
    generated_queries: list  # 生成的原始查询
    user_confirmed_queries: list  # 用户确认/修改后的查询
    awaiting_user_confirmation: bool  # 是否等待用户确认
    user_confirmation_received: bool  # 是否已收到用户确认
    # 其他状态字段，方便节点各自返回需要的JSON内容。
    content_quality: ContentQualityState
    fact_verification: FactVerificationState
    reflection: ReflectionState
    translation: TranslationState
    relevance_assessment: RelevanceState
    summary_optimization: SummaryOptimizationState
    quality_enhanced_summary: str
    verification_report: str
    final_confidence_score: float