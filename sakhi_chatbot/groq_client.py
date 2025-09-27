"""Client for interacting with the Groq chat completions API.

This module keeps all Groq specific logic in one place so that the rest
of the application can call a simple `GroqChatClient` abstraction without
worrying about HTTP details or system prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import json
import os
import time

import requests


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class GroqMessage:
    """Represents a chat style message that can be sent to Groq."""
    role: str
    content: str


class GroqAPIError(RuntimeError):
    """Raised when the Groq API returns an unexpected response."""


class GroqChatClient:
    """Thin wrapper around the Groq REST API used for chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_GROQ_MODEL,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required to call the Groq API."
            )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _build_payload(
        self,
        messages: Iterable[GroqMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [message.__dict__ for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def complete(
        self,
        messages: Iterable[GroqMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 1.5,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """Send chat completion request to Groq and return the parsed JSON."""

        payload = self._build_payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format:
            payload["response_format"] = response_format
        url = "https://api.groq.com/openai/v1/chat/completions"

        last_error: Optional[str] = None
        for attempt in range(retry_attempts + 1):
            try:
                response = self._session.post(
                    url, json=payload, timeout=self.timeout_seconds
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    try:
                        body = exc.response.json()
                    except ValueError:
                        body = exc.response.text
                    detail = f"{exc}. Response details: {body}"
                else:
                    detail = str(exc)
                last_error = detail
                if attempt == retry_attempts:
                    break
                time.sleep(retry_backoff_seconds * (attempt + 1))
        raise GroqAPIError(f"Groq completion failed: {last_error}")

    def structured_complete(
        self,
        system_prompt: str,
        conversation: List[GroqMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.4,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        messages = [GroqMessage(role="system", content=system_prompt)] + conversation
        return self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    @staticmethod
    def extract_message_text(response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise GroqAPIError("Groq response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise GroqAPIError("Groq response missing message content")
        return content
