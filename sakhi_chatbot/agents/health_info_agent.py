"""Agent that surfaces reliable health guidance."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import textwrap

import requests

from ..external_sources import ExternalAdvice, default_sources


@dataclass
class LocalAdvice:
    """Representation of cached health guidance entries."""

    title: str
    summary: str
    source: str
    url: str

    def render(self) -> str:
        summary = textwrap.fill(self.summary.strip(), width=90)
        parts = [self.title.strip()] if self.title else []
        if summary:
            parts.append(summary)
        if self.source:
            parts.append(f"स्रोत: {self.source}")
        if self.url:
            parts.append(f"अधिक जानकारी: {self.url}")
        return "\n".join(parts)


class HealthKnowledgeAgent:
    """Combine cached guidance with remote API lookups when available."""

    def __init__(self, knowledge_base_path: Path) -> None:
        self.knowledge_base_path = knowledge_base_path
        self._local: Dict[str, LocalAdvice] = {}
        self._load_local()
        self._session = requests.Session()
        self._api_url = os.getenv("HEALTH_INFO_API", "").strip() or None
        self._external_sources = default_sources()

    def _load_local(self) -> None:
        if not self.knowledge_base_path.exists():
            self._local = {}
            return
        try:
            data = json.loads(self.knowledge_base_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._local = {}
            return
        parsed: Dict[str, LocalAdvice] = {}
        for key, entry in data.items():
            parsed[key.lower()] = LocalAdvice(
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
        sections = []
        for advice in self._fetch_external(key):
            sections.append(advice.render())
        local = self._local.get(key)
        if local:
            sections.append(local.render())
        if sections:
            return "\n\n".join(sections)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_external(self, topic: str) -> Iterable[ExternalAdvice]:
        for source in self._external_sources:
            advice = source.fetch(topic)
            if advice:
                yield advice
        remote_env = self._fetch_env_api(topic)
        if remote_env:
            yield remote_env

    def _fetch_env_api(self, topic: str) -> Optional[ExternalAdvice]:
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
        summary = payload.get("summary") or payload.get("advice") or ""
        if not str(summary).strip():
            return None
        return ExternalAdvice(
            title=str(payload.get("title", topic.title())),
            summary=str(summary),
            source=str(payload.get("source", "")),
            url=str(payload.get("url", "")),
        )


__all__ = ["HealthKnowledgeAgent"]
