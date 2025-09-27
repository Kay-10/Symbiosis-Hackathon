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

_ROMANIZED_HINDI_TOKENS = {
    "mai",
    "main",
    "meri",
    "mera",
    "nahin",
    "nahi",
    "haan",
    "kyu",
    "kyun",
    "kyon",
    "dard",
    "bimar",
    "thik",
    "theek",
    "sar",
    "sir",
    "pet",
    "khana",
}


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
            inferred = self._heuristic_language(user_text)
            return LanguageDecision(
                inferred,
                "langdetect not installed; heuristic applied",
            )
        try:
            lang_code = detect(user_text)
        except LangDetectException:
            inferred = self._heuristic_language(user_text)
            return LanguageDecision(inferred, "Detection error; heuristic applied")

        mapped = SUPPORTED_LANGUAGES.get(lang_code)
        if mapped:
            return LanguageDecision(mapped, f"Detected {lang_code}")

        inferred = self._heuristic_language(user_text)
        if inferred != self.default_language:
            return LanguageDecision(inferred, f"Heuristic override for romanized text from {lang_code}")
        return LanguageDecision(self.default_language, f"Unsupported {lang_code}")

    def _heuristic_language(self, user_text: str) -> str:
        lowered = user_text.lower()
        tokens = lowered.split()
        score_hi = sum(1 for token in tokens if token in _ROMANIZED_HINDI_TOKENS)
        if score_hi >= 2:
            return SUPPORTED_LANGUAGES["hi"]
        return self.default_language


__all__ = ["LanguageRouter", "LanguageDecision", "SUPPORTED_LANGUAGES"]
