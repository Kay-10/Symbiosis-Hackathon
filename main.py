from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sakhi_chatbot.assistant import AssistantTurnResult, SakhiAssistant
from sakhi_chatbot.groq_client import GroqAPIError, GroqChatClient, DEFAULT_GROQ_MODEL
from sakhi_chatbot.memory import ConversationMemory

EXIT_COMMANDS = {"quit", "exit", "bye"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sakhi conversational healthcare assistant")
    parser.add_argument(
        "--mode",
        choices=["text", "voice"],
        default="text",
        help="Run in text (terminal) mode or voice mode",
    )
    parser.add_argument(
        "--history",
        default=".storage/chat_history.json",
        help="Path to persist conversation history",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Start with a clean history file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GROQ_MODEL,
        help="Groq model identifier to use for chat completions",
    )
    return parser.parse_args()


def build_memory(path: Path, clear: bool) -> ConversationMemory:
    if clear and path.exists():
        path.unlink()
    return ConversationMemory(storage_path=path)


def build_assistant(model: str, memory: ConversationMemory) -> SakhiAssistant:
    try:
        groq_client = GroqChatClient(model=model)
    except ValueError:
        print("GROQ_API_KEY not set. Export GROQ_API_KEY before running Sakhi.")
        sys.exit(1)
    return SakhiAssistant(groq_client=groq_client, memory=memory)


def render_result(result: AssistantTurnResult, *, voice=None) -> None:
    print(f"Sakhi: {result.message}")
    if voice:
        voice.say(result.message)
    if result.safety_note:
        print(f"Safety: {result.safety_note}")
        if result.encourage_doctor and voice:
            voice.say(result.safety_note)
    if result.agent_name != "NONE":
        print(
            "[Agent]",
            result.agent_name,
            "inputs:",
            result.agent_inputs or "{}",
        )


def run_text_mode(assistant: SakhiAssistant) -> None:
    print("Sakhi assistant ready. Type your question (or 'quit' to exit).\n")
    while True:
        try:
            raw = input("You: ")
        except EOFError:
            print()
            break
        message = raw.strip()
        if not message:
            continue
        if message.lower() in EXIT_COMMANDS:
            break
        try:
            result = assistant.handle_user_message(
                message,
                on_intermediate=lambda msg, _lang: print(f"Sakhi: {msg}"),
            )
        except GroqAPIError as exc:
            print(f"Sakhi encountered an error: {exc}")
            continue
        render_result(result)
    print("Goodbye from Sakhi.")


def run_voice_mode(assistant: SakhiAssistant) -> None:
    try:
        from sakhi_chatbot.voice_io import VoiceConfig, VoiceInterface
    except ImportError as exc:
        print("Voice dependencies missing. Install speechrecognition and pyttsx3.")
        sys.exit(1)

    voice = VoiceInterface()
    config = VoiceConfig()
    print("Voice mode active. Say 'stop' or 'quit' to finish.")
    while True:
        transcript = voice.listen_once(config)
        if not transcript:
            print("I couldn't hear that clearly. Let's try again.")
            continue
        print(f"You said: {transcript}")
        normalized = transcript.lower().strip()
        if any(cmd in normalized for cmd in EXIT_COMMANDS) or "stop" in normalized:
            break
        try:
            def voice_wait(msg: str, _lang: str) -> None:
                print(f"Sakhi: {msg}")
                voice.say(msg)

            result = assistant.handle_user_message(
                transcript,
                on_intermediate=voice_wait,
            )
        except GroqAPIError as exc:
            print(f"Sakhi encountered an error: {exc}")
            continue
        render_result(result, voice=voice)
    print("Voice session ended. Goodbye from Sakhi.")


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    memory = build_memory(history_path, args.clear_history)
    assistant = build_assistant(args.model, memory)

    if args.mode == "voice":
        run_voice_mode(assistant)
    else:
        run_text_mode(assistant)


if __name__ == "__main__":
    main()
