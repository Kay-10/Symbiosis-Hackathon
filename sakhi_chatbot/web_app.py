from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import edge_tts

genai = None
try:  # pragma: no cover - optional dependency check
    genai = importlib.import_module("google.generativeai")
except ModuleNotFoundError:
    genai = None
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .assistant import SakhiAssistant
from .groq_client import GroqChatClient, GroqAPIError
from .language import LanguageRouter
from .memory import ConversationMemory

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_DIR = WEB_DIR


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    language: str
    encourage_doctor: bool
    history: List[Dict[str, str]]


@dataclass
class SessionState:
    assistant: SakhiAssistant
    lock: asyncio.Lock


class SessionManager:
    def __init__(self) -> None:
        self._storage_root = Path(".storage/web_sessions")
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._client = GroqChatClient()
        self._sessions: Dict[str, SessionState] = {}
        self._sessions_lock = Lock()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        state = self._build_session(session_id)
        with self._sessions_lock:
            self._sessions[session_id] = state
        return session_id

    def get(self, session_id: str) -> SessionState:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state:
            return state
        memory_path = self._storage_root / f"{session_id}.json"
        if not memory_path.exists():
            raise KeyError(session_id)
        state = self._build_session(session_id)
        with self._sessions_lock:
            self._sessions[session_id] = state
        return state

    def _build_session(self, session_id: str) -> SessionState:
        memory_path = self._storage_root / f"{session_id}.json"
        memory = ConversationMemory(storage_path=memory_path)
        assistant = SakhiAssistant(groq_client=self._client, memory=memory)
        return SessionState(assistant=assistant, lock=asyncio.Lock())

    def history(self, session: SessionState) -> List[Dict[str, str]]:
        return [
            {"role": turn.role, "content": turn.content}
            for turn in session.assistant.memory.history
        ]


session_manager = SessionManager()
language_router = LanguageRouter()
VOICE_MAP = {
    "hi-IN": "hi-IN-SwaraNeural",
    "mr-IN": "mr-IN-AarohiNeural",
    "en-IN": "en-IN-NeerjaNeural",
    "bn-IN": "bn-IN-TanishaaNeural",
    "ta-IN": "ta-IN-PriyaNeural",
    "te-IN": "te-IN-ShrutiNeural",
    "ml-IN": "ml-IN-SobhanaNeural",
    "gu-IN": "gu-IN-DhwaniNeural",
    "kn-IN": "kn-IN-SapnaNeural",
    "pa-IN": "pa-IN-AmarDeepNeural",
    "ur-IN": "ur-IN-GulNeural",
}
DEFAULT_VOICE = "en-IN-NeerjaNeural"
app = FastAPI(title="Sakhi Voice Companion", version="1.0.0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash")
GEMINI_MODEL = None
if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/session")
async def create_session() -> Dict[str, str]:
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.get("/api/history")
async def get_history(session_id: str) -> Dict[str, List[Dict[str, str]]]:
    try:
        session = session_manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": session_manager.history(session)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        session = session_manager.get(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    async with session.lock:
        try:
            result = await run_in_threadpool(
                session.assistant.handle_user_message,
                request.message.strip(),
            )
        except GroqAPIError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    history = session_manager.history(session)
    return ChatResponse(
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=result.message,
        language=result.language,
        encourage_doctor=result.encourage_doctor,
        history=history,
    )


class SpeakRequest(BaseModel):
    text: str


async def _generate_tts_audio(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
    return bytes(audio_bytes)


@app.post("/api/speak")
async def speak(payload: SpeakRequest) -> StreamingResponse:
    message = payload.text.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty text")
    language_code = language_router.detect_language(message).language_code
    voice = VOICE_MAP.get(language_code, DEFAULT_VOICE)
    try:
        audio_bytes = await _generate_tts_audio(message, voice)
    except Exception as exc:  # pragma: no cover - network/remote errors
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {exc}")
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")


@app.post("/api/transcribe")
async def transcribe_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
) -> Dict[str, str]:
    # Ensure session exists so clients can't call with stale session IDs
    try:
        session_manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio stream")

    try:
        text, lang = await run_in_threadpool(_transcribe_bytes, data)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not text:
        raise HTTPException(status_code=422, detail="Could not understand audio")
    return {"text": text, "language": lang}


def _transcribe_bytes(data: bytes) -> tuple[Optional[str], Optional[str]]:
    if not GEMINI_MODEL:
        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini transcription is not configured. Set GEMINI_API_KEY.")
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. Run `pip install google-generativeai`."
            )
        raise RuntimeError("Gemini transcription model could not be initialised.")

    audio_base64 = base64.b64encode(data).decode("utf-8")
    prompt_parts = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Transcribe the provided audio recording. "
                        "The speaker may use Hindi, Marathi, or English. "
                        "Return only the words they spoke with no extra commentary."
                    )
                },
                {
                    "mime_type": "audio/wav",
                    "data": audio_base64,
                },
            ],
        }
    ]

    try:
        response = GEMINI_MODEL.generate_content(prompt_parts, request_options={"timeout": 60})
    except Exception as exc:  # pragma: no cover - network/API failures
        raise RuntimeError(f"Gemini transcription failed: {exc}") from exc

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        return None, None

    try:
        language_code = language_router.detect_language(text).language_code
    except Exception:
        language_code = None

    return text, language_code
