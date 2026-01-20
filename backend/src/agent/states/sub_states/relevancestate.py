from typing import TypedDict


class RelevanceState(TypedDict):
    relevance_score: float
    key_topics_covered: list[str]
    missing_topics: list[str]
    content_alignment: str