"""Agent that provides vetted health guidance snippets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import textwrap


@dataclass
class HealthArticle:
    title: str
    summary: str
    source: str
    url: str

    def render(self) -> str:
        wrapped = textwrap.fill(self.summary, width=90)
        return f"{self.title}\nSource: {self.source}\nLink: {self.url}\n{wrapped}"


class HealthKnowledgeAgent:
    """Fetch health facts from a curated knowledge base."""

    def __init__(self, knowledge_base_path: Path) -> None:
        self.knowledge_base_path = knowledge_base_path
        self._cache: Dict[str, HealthArticle] = {}
        self._load_data()

    def _load_data(self) -> None:
        if not self.knowledge_base_path.exists():
            self._cache = {}
            return
        try:
            data = json.loads(self.knowledge_base_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._cache = {}
            return
        for topic, payload in data.items():
            self._cache[topic.lower()] = HealthArticle(
                title=payload.get("title", topic.title()),
                summary=payload.get("summary", ""),
                source=payload.get("source", ""),
                url=payload.get("url", ""),
            )

    def fetch(self, topic: str) -> Optional[str]:
        entry = self._cache.get(topic.lower())
        if not entry:
            return None
        return entry.render()


__all__ = ["HealthKnowledgeAgent"]
