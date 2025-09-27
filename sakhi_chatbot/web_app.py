from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import speech_recognition as sr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .assistant import SakhiAssistant
from .groq_client import GroqChatClient, GroqAPIError
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
app = FastAPI(title="Sakhi Voice Companion", version="1.0.0")

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

    text, lang = await run_in_threadpool(_transcribe_bytes, data)
    if not text:
        raise HTTPException(status_code=422, detail="Could not understand audio")
    return {"text": text, "language": lang}


def _transcribe_bytes(data: bytes) -> tuple[Optional[str], Optional[str]]:
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(data)) as source:
        audio = recognizer.record(source)
    for lang in ("hi-IN", "en-IN"):
        try:
            text = recognizer.recognize_google(audio, language=lang)
            return text, lang
        except sr.UnknownValueError:
            continue
        except sr.RequestError:
            break
    try:
        text = recognizer.recognize_google(audio)
        return text, None
    except sr.UnknownValueError:
        return None, None
    except sr.RequestError:
        return None, None
