"""Local directory agent providing nearby healthcare contacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LocalResource:
    name: str
    address: str
    phone: str
    hours: str
    notes: str

    def render(self) -> str:
        segments = [self.name]
        if self.address:
            segments.append(self.address)
        if self.phone:
            segments.append(f"फोन: {self.phone}")
        if self.hours:
            segments.append(f"समय: {self.hours}")
        if self.notes:
            segments.append(f"नोट: {self.notes}")
        return " | ".join(segments)


class LocalDirectoryAgent:
    """Lookup agent returning formatted clinic information by PIN code."""

    def __init__(self, directory_path: Path) -> None:
        self.directory_path = directory_path
        self._cache: Dict[str, List[LocalResource]] = {}
        self._load()

    def _load(self) -> None:
        if not self.directory_path.exists():
            self._cache = {}
            return
        try:
            raw = self.directory_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            self._cache = {}
            return
        parsed: Dict[str, List[LocalResource]] = {}
        for code, entries in data.items():
            resources: List[LocalResource] = []
            for entry in entries:
                resources.append(
                    LocalResource(
                        name=entry.get("name", ""),
                        address=entry.get("address", ""),
                        phone=entry.get("phone", ""),
                        hours=entry.get("hours", ""),
                        notes=entry.get("notes", ""),
                    )
                )
            parsed[code] = resources
        self._cache = parsed

    def lookup(self, pincode: str) -> List[LocalResource]:
        if not pincode:
            return []
        return list(self._cache.get(str(pincode).strip(), []))

    def formatted_directory(self, pincode: str) -> Optional[str]:
        records = self.lookup(pincode)
        if not records:
            return None
        lines = [resource.render() for resource in records]
        body = "\n".join(f"• {line}" for line in lines)
        return f"PIN {pincode} के पास के भरोसेमंद विकल्प:\n{body}"


__all__ = ["LocalDirectoryAgent", "LocalResource"]
