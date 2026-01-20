from typing import TypedDict


class TranslationState(TypedDict):
    original_language: str
    target_language: str
    translated_content: str
    translation_quality: str