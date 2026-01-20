from typing import TypedDict


class SummaryOptimizationState(TypedDict):
    optimized_summary: str
    key_insights: list[str]
    actionable_items: list[str]
    confidence_level: str