"""Simple persisted conversation memory for follow-up queries."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    """Stores chat turns in memory and flushes them to JSON for continuity."""

    storage_path: Path
    history: List[ChatTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.storage_path.exists():
            self._load()

    def append(self, role: str, content: str) -> None:
        self.history.append(ChatTurn(role=role, content=content))
        self._persist()

    def extend(self, turns: Iterable[ChatTurn]) -> None:
        self.history.extend(turns)
        self._persist()

    def as_groq_messages(self) -> List[Dict[str, str]]:
        return [turn.__dict__ for turn in self.history]

    # --- internal helpers -------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in data:
            role = item.get("role")
            content = item.get("content")
            if not role or not content:
                continue
            self.history.append(ChatTurn(role=role, content=content))

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [turn.__dict__ for turn in self.history]
        self.storage_path.write_text(json.dumps(data, ensure_ascii=True, indent=2))


def default_memory(base_dir: str = ".storage") -> ConversationMemory:
    storage_root = Path(base_dir)
    storage_root.mkdir(parents=True, exist_ok=True)
    return ConversationMemory(storage_path=storage_root / "chat_history.json")
