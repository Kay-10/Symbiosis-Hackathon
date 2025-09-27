"""Utilities for capturing speech input and producing spoken output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import speech_recognition as sr

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None  # type: ignore


@dataclass
class VoiceConfig:
    microphone_index: Optional[int] = None
    noise_calibration_duration: float = 1.5
    phrase_time_limit: Optional[int] = 8
    primary_locale: str = "en-IN"
    fallback_locale: str = "hi-IN"


class VoiceInterface:
    """Wraps SpeechRecognition and optional text-to-speech playback."""

    def __init__(self, recognizer: Optional[sr.Recognizer] = None) -> None:
        self.recognizer = recognizer or sr.Recognizer()
        self.tts_engine = pyttsx3.init() if pyttsx3 else None

    def listen_once(self, config: VoiceConfig) -> Optional[str]:
        microphone = self._initialize_microphone(config)
        if not microphone:
            return None

        with microphone as source:
            self.recognizer.energy_threshold = 4000
            self.recognizer.adjust_for_ambient_noise(
                source, duration=config.noise_calibration_duration
            )
            print("Listening...")
            audio = self.recognizer.listen(
                source,
                timeout=None,
                phrase_time_limit=config.phrase_time_limit,
            )
        try:
            return self.recognizer.recognize_google(audio, language=config.primary_locale)
        except sr.UnknownValueError:
            if config.fallback_locale:
                print(f"Retrying speech recognition with {config.fallback_locale}")
                try:
                    return self.recognizer.recognize_google(
                        audio, language=config.fallback_locale
                    )
                except (sr.UnknownValueError, sr.RequestError):
                    return None
            return None
        except sr.RequestError:
            return None

    def say(self, text: str) -> None:
        if not text.strip():
            return
        if self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        else:
            print("[TTS unavailable]", text)

    def _initialize_microphone(self, config: VoiceConfig) -> Optional[sr.Microphone]:
        try:
            if config.microphone_index is None:
                return sr.Microphone()
            return sr.Microphone(device_index=config.microphone_index)
        except Exception as exc:  # pragma: no cover - hardware specific branch
            print(f"Failed to initialize microphone: {exc}")
            return None
