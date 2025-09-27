from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sakhi_chatbot.assistant import AssistantTurnResult, SakhiAssistant
from sakhi_chatbot.groq_client import DEFAULT_GROQ_MODEL, GroqAPIError, GroqChatClient
from sakhi_chatbot.memory import ConversationMemory

EXIT_COMMANDS = {"quit", "exit", "bye"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sakhi conversational healthcare assistant (text CLI)")
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



def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    memory = build_memory(history_path, args.clear_history)
    assistant = build_assistant(args.model, memory)

    run_text_mode(assistant)


if __name__ == "__main__":
    main()
