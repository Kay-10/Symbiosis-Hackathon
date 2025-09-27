"""Agent responsible for returning local healthcare assistance details."""
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

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "address": self.address,
            "phone": self.phone,
            "hours": self.hours,
            "notes": self.notes,
        }


class LocalDirectoryAgent:
    """Lookup agent using a small curated directory keyed by PIN code."""

    def __init__(self, directory_path: Path) -> None:
        self.directory_path = directory_path
        self._cache: Dict[str, List[LocalResource]] = {}
        self._load_directory()

    def _load_directory(self) -> None:
        if not self.directory_path.exists():
            self._cache = {}
            return
        try:
            data = json.loads(self.directory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._cache = {}
            return
        for pincode, entries in data.items():
            resources = []
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
            self._cache[pincode] = resources

    def lookup(self, pincode: str) -> List[Dict[str, str]]:
        resources = self._cache.get(str(pincode).strip())
        if not resources:
            return []
        return [resource.to_dict() for resource in resources]


__all__ = ["LocalDirectoryAgent"]
