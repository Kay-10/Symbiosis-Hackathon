"""External health information connectors for Sakhi."""
from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import requests

try:  # Optional dependency used for simple HTML extraction.
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class ExternalAdvice:
    """Aggregated advice snippet sourced from a reliable provider."""

    title: str
    summary: str
    source: str
    url: str

    def render(self) -> str:
        body = textwrap.fill(self.summary.strip(), width=90)
        segments = [self.title.strip()] if self.title else []
        if body:
            segments.append(body)
        if self.source:
            segments.append(f"स्रोत: {self.source}")
        if self.url:
            segments.append(f"अधिक जानकारी: {self.url}")
        return "\n".join(segments)


class ExternalSource(Protocol):
    """Protocol for remote health sources."""

    def fetch(self, topic: str) -> Optional[ExternalAdvice]:  # pragma: no cover - interface
        ...


class MyHealthFinderSource:
    """US Health & Human Services MyHealthfinder API (open, no key)."""

    API_URL = "https://health.gov/myhealthfinder/api/v3/topicsearch.json"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, topic: str) -> Optional[ExternalAdvice]:
        params = {"keyword": topic.lower()}
        try:
            response = self._session.get(self.API_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure branch
            LOGGER.debug("MyHealthFinder request failed: %s", exc)
            return None
        try:
            payload = response.json()
        except json.JSONDecodeError:
            LOGGER.debug("MyHealthFinder returned non-JSON response")
            return None
        try:
            items = payload["Result"]["Resources"]["Resource"]
        except (KeyError, TypeError):
            return None
        if not items:
            return None
        first = items[0]
        title = first.get("Title", "Health Advice")
        url = first.get("AccessibleVersion", "")
        sections = first.get("Sections", {}).get("section", [])
        summary_parts = []
        for section in sections:
            content = section.get("Description")
            if content:
                summary_parts.append(content)
        if not summary_parts:
            summary = first.get("Description") or ""
        else:
            summary = "\n".join(summary_parts)
        return ExternalAdvice(
            title=title,
            summary=summary,
            source="US Department of Health & Human Services",
            url=url,
        )


class NHPHtmlSource:
    """Minimal scraper for National Health Portal topics."""

    BASE_URL = "https://www.nhp.gov.in"
    TOPIC_SLUGS: Dict[str, str] = {
        "anaemia": "/disease/gynaecology-and-obstetrics/anaemia",
        "diarrhoea": "/disease/gastrointestinal/diarrhoea",
        "malaria": "/disease/infectious-parasitic/malaria",
        "dengue": "/disease/infectious-parasitic/dengue",
        "headache": "/disease/neurology/headache",
        "hypertension": "/disease/cardiovascular/hypertension",
    }

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, topic: str) -> Optional[ExternalAdvice]:
        if BeautifulSoup is None:
            return None
        slug = self.TOPIC_SLUGS.get(topic.lower())
        if not slug:
            return None
        url = f"{self.BASE_URL}{slug}"
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure branch
            LOGGER.debug("NHP request failed: %s", exc)
            return None
        soup = BeautifulSoup(response.text, "html.parser")  # type: ignore[arg-type]
        article = soup.find("div", class_="field-item") or soup
        paragraphs = [p.get_text(strip=True) for p in article.find_all("p")][:3]
        summary = " ".join(paragraphs).strip()
        if not summary:
            return None
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else topic.title()
        return ExternalAdvice(
            title=title,
            summary=summary,
            source="National Health Portal of India",
            url=url,
        )


class UNICEFIndiaSource:
    """Scrapes curated UNICEF India health pages."""

    TOPIC_SLUGS: Dict[str, str] = {
        "breastfeeding": "https://www.unicef.org/india/what-we-do/breastfeeding",
        "maternal health": "https://www.unicef.org/india/what-we-do/maternal-health",
        "nutrition": "https://www.unicef.org/india/what-we-do/nutrition",
    }

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def fetch(self, topic: str) -> Optional[ExternalAdvice]:
        if BeautifulSoup is None:
            return None
        url = self.TOPIC_SLUGS.get(topic.lower())
        if not url:
            return None
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover
            LOGGER.debug("UNICEF request failed: %s", exc)
            return None
        soup = BeautifulSoup(response.text, "html.parser")  # type: ignore[arg-type]
        paragraphs = [p.get_text(strip=True) for p in soup.select("article p")][:3]
        summary = " ".join(paragraphs).strip()
        if not summary:
            return None
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "UNICEF Guidance"
        return ExternalAdvice(
            title=title,
            summary=summary,
            source="UNICEF India",
            url=url,
        )


def default_sources() -> List[ExternalSource]:
    """Return the default set of external connectors."""

    return [
        MyHealthFinderSource(),
        NHPHtmlSource(),
        UNICEFIndiaSource(),
    ]


__all__ = [
    "ExternalAdvice",
    "ExternalSource",
    "MyHealthFinderSource",
    "NHPHtmlSource",
    "UNICEFIndiaSource",
    "default_sources",
]
