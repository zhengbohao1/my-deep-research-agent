from typing import TypedDict


class FactVerificationState(TypedDict):
    verified_facts: list[dict]
    disputed_claims: list[dict]
    verification_sources: list[str]
    confidence_score: float