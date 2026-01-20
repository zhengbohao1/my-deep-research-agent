from typing import TypedDict


class ContentQualityState(TypedDict):
    quality_score: float
    reliability_assessment: str
    content_gaps: list[str]
    improvement_suggestions: list[str]