"""Agent that surfaces reliable health guidance."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import textwrap

import requests


@dataclass
class HealthArticle:
    title: str
    summary: str
    source: str
    url: str

    def render(self) -> str:
        summary = textwrap.fill(self.summary, width=90)
        lines = [self.title]
        if self.source:
            lines.append(f"स्रोत: {self.source}")
        if self.url:
            lines.append(f"अधिक जानकारी: {self.url}")
        lines.append(summary)
        return "\n".join(lines)


class HealthKnowledgeAgent:
    """Combine cached knowledge with optional trusted API lookups."""

    def __init__(self, knowledge_base_path: Path) -> None:
        self.knowledge_base_path = knowledge_base_path
        self._local: Dict[str, HealthArticle] = {}
        self._load_local()
        self._session = requests.Session()
        self._api_url = os.getenv("HEALTH_INFO_API")

    def _load_local(self) -> None:
        if not self.knowledge_base_path.exists():
            self._local = {}
            return
        try:
            raw = self.knowledge_base_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            self._local = {}
            return
        parsed: Dict[str, HealthArticle] = {}
        for key, entry in data.items():
            parsed[key.lower()] = HealthArticle(
                title=entry.get("title", key.title()),
                summary=entry.get("summary", ""),
                source=entry.get("source", ""),
                url=entry.get("url", ""),
            )
        self._local = parsed

    def fetch(self, topic: str) -> Optional[str]:
        key = topic.strip().lower()
        if not key:
            return None
        article = self._local.get(key)
        remote = self._fetch_remote_article(key)
        combined = []
        if article:
            combined.append(article.render())
        if remote:
            combined.append(remote)
        if not combined:
            return None
        return "\n\n".join(combined)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_remote_article(self, topic: str) -> Optional[str]:
        if not self._api_url:
            return None
        try:
            response = self._session.get(
                self._api_url,
                params={"topic": topic},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        title = payload.get("title")
        advice = payload.get("advice")
        source = payload.get("source")
        if not advice:
            return None
        summary = textwrap.fill(str(advice), width=90)
        details = []
        if title:
            details.append(str(title))
        details.append(summary)
        if source:
            details.append(f"स्रोत: {source}")
        link = payload.get("url")
        if link:
            details.append(f"अधिक जानकारी: {link}")
        return "\n".join(details)


__all__ = ["HealthKnowledgeAgent"]
