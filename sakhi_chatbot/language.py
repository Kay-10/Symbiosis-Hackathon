"""Utility helpers for detecting and routing languages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover - optional runtime dependency
    detect = None  # type: ignore
    DetectorFactory = None  # type: ignore
    LangDetectException = Exception  # type: ignore


SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "pa": "pa-IN",
    "ur": "ur-IN",
}

FALLBACK_LANGUAGE = "en-IN"


@dataclass(frozen=True)
class LanguageDecision:
    language_code: str
    reason: str


class LanguageRouter:
    """Choose the best BCP47 locale based on user input."""

    def __init__(self, default_language: str = FALLBACK_LANGUAGE) -> None:
        self.default_language = default_language

    def detect_language(self, user_text: str) -> LanguageDecision:
        if not user_text.strip():
            return LanguageDecision(self.default_language, "Empty input")

        if detect is None:
            return LanguageDecision(
                self.default_language,
                "langdetect not installed; falling back to default",
            )
        try:
            lang_code = detect(user_text)
        except LangDetectException:
            return LanguageDecision(self.default_language, "Detection error")

        mapped = SUPPORTED_LANGUAGES.get(lang_code)
        if mapped:
            return LanguageDecision(mapped, f"Detected {lang_code}")
        return LanguageDecision(self.default_language, f"Unsupported {lang_code}")


__all__ = ["LanguageRouter", "LanguageDecision", "SUPPORTED_LANGUAGES"]
