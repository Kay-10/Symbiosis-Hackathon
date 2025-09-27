"""High level orchestration for the Sakhi healthcare assistant."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .agents.health_info_agent import HealthKnowledgeAgent
from .agents.local_info_agent import LocalDirectoryAgent
from .groq_client import GroqChatClient, GroqMessage, GroqAPIError
from .language import LanguageRouter
from .memory import ConversationMemory

PRIMARY_SYSTEM_PROMPT = """
You are Sakhi, a trusted community health guide supporting rural women in India.
You always respond with JSON using this schema: {"language": string, "assistant_reply": string, "agent": {"name": string, "inputs": object}, "encourage_doctor": boolean, "safety_note": string}.
Detect the user's language from the latest message and use the closest available Indian language locale (BCP-47).
If you receive an extra system message prefixed with LANGUAGE_HINT::, favour that code when picking the language.
For serious symptoms, set encourage_doctor=true and craft assistant_reply that urges the user to seek professional help.
The agent name must be one of NONE, LOCAL_DIRECTORY, or HEALTH_KNOWLEDGE.
If an agent other than NONE is required, make assistant_reply a short message asking the user to wait while you collect information.
Inputs for LOCAL_DIRECTORY should include a "pincode" field if available.
Inputs for HEALTH_KNOWLEDGE should include a "topic" string that maps to the curated knowledge base topics: anaemia, prenatal, breastfeeding.
Keep tone respectful, culturally sensitive, and always remind users to see a doctor for alarming warning signs.
""".strip()

AGENT_SUMMARY_PROMPT_TEMPLATE = (
    "You are Sakhi, continuing the same conversation in {language}. Summarise the "
    "agent findings below in the user's language, include relevant cautions, and "
    "encourage medical consultation when warranted. Be concise and empathetic."
)

WAITING_TRANSLATIONS: Dict[str, str] = {
    "hi-IN": "कृपया थोड़ा इंतज़ार करें, मैं जानकारी जुटा रही हूँ।",
    "bn-IN": "অনুগ্রহ করে একটু অপেক্ষা করুন, আমি তথ্য সংগ্রহ করছি।",
    "te-IN": "దయచేసి కాసేపు వేచి ఉండండి, నేనే సమాచారం తెస్తున్నాను.",
    "ta-IN": "தயவு செய்து ஒரு நிமிடம் காத்திருக்கவும், தகவலை கொண்டுவருகிறேன்.",
    "ml-IN": "ദയവായി ഒരു നിമിഷം കാത്തിരിക്കൂ, ഞാൻ വിവരങ്ങൾ ശേഖരിക്കുന്നു.",
    "mr-IN": "कृपया थोडा वेळ थांबा, मी माहिती गोळा करत आहे.",
    "gu-IN": "મહેરબાની કરીને થોડી રાહ જુઓ, હું માહિતી મેળવી રહી છું.",
    "kn-IN": "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಕಾಯಿರಿ, ನಾನು ಮಾಹಿತಿಯನ್ನು ಸಂಗ್ರಹಿಸುತ್ತಿದ್ದೇನೆ.",
    "pa-IN": "ਕਿਰਪਾ ਕਰਕੇ ਥੋੜ੍ਹਾ ਇੰਤਜ਼ਾਰ ਕਰੋ, ਮੈਂ ਜਾਣਕਾਰੀ ਲੈ ਰਹੀ ਹਾਂ।",
    "ur-IN": "براہ کرم کچھ دیر انتظار کریں، میں معلومات جمع کر رہی ہوں۔",
}


@dataclass
class AgentDirective:
    name: str = "NONE"
    inputs: Dict[str, str] = field(default_factory=dict)


@dataclass
class GroqDirective:
    language: str
    assistant_reply: str
    agent: AgentDirective
    encourage_doctor: bool
    safety_note: str


@dataclass
class AssistantTurnResult:
    language: str
    message: str
    agent_name: str
    agent_inputs: Dict[str, str]
    safety_note: str
    encourage_doctor: bool
    agent_output: Optional[str] = None
    intermediate_message: Optional[str] = None


class SakhiAssistant:
    """Coordinate language routing, Groq reasoning, and specialist agents."""

    def __init__(
        self,
        groq_client: GroqChatClient,
        memory: ConversationMemory,
        *,
        language_router: Optional[LanguageRouter] = None,
        local_agent: Optional[LocalDirectoryAgent] = None,
        health_agent: Optional[HealthKnowledgeAgent] = None,
    ) -> None:
        self.groq_client = groq_client
        self.memory = memory
        self.language_router = language_router or LanguageRouter()
        self.local_agent = local_agent or LocalDirectoryAgent(
            Path(__file__).resolve().parent / "data" / "local_health_directory.json"
        )
        self.health_agent = health_agent or HealthKnowledgeAgent(
            Path(__file__).resolve().parent / "data" / "health_knowledge_base.json"
        )

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------
    def handle_user_message(
        self,
        user_text: str,
        *,
        on_intermediate: Optional[Callable[[str, str], None]] = None,
    ) -> AssistantTurnResult:
        lang_decision = self.language_router.detect_language(user_text)
        self.memory.append("user", user_text)

        conversation = self._memory_as_messages()
        directive = self._query_groq(conversation, language_hint=lang_decision.language_code)

        self.memory.append("assistant", directive.assistant_reply)

        if directive.agent.name == "NONE":
            return AssistantTurnResult(
                language=directive.language,
                message=directive.assistant_reply,
                agent_name="NONE",
                agent_inputs={},
                safety_note=directive.safety_note,
                encourage_doctor=directive.encourage_doctor,
                intermediate_message=None,
            )

        if on_intermediate:
            on_intermediate(directive.assistant_reply, directive.language)

        agent_result = self._run_agent(directive.agent)
        if not agent_result:
            agent_result = "No trusted information was found. Encourage the user to contact a local doctor or ASHA worker."

        agent_summary = self._summarise_agent_response(
            language=directive.language,
            agent_name=directive.agent.name,
            agent_inputs=directive.agent.inputs,
            agent_output=agent_result,
            safety_note=directive.safety_note,
            encourage_doctor=directive.encourage_doctor,
        )

        self.memory.append("assistant", agent_summary)
        return AssistantTurnResult(
            language=directive.language,
            message=agent_summary,
            agent_name=directive.agent.name,
            agent_inputs=directive.agent.inputs,
            safety_note=directive.safety_note,
            encourage_doctor=directive.encourage_doctor,
            agent_output=agent_result,
            intermediate_message=directive.assistant_reply,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_groq(
        self,
        conversation: List[GroqMessage],
        *,
        language_hint: Optional[str] = None,
    ) -> GroqDirective:
        convo = list(conversation)
        if language_hint:
            convo.append(GroqMessage(role="system", content=f"LANGUAGE_HINT::{language_hint}"))
        response = self.groq_client.structured_complete(
            PRIMARY_SYSTEM_PROMPT,
            convo,
        )
        raw_text = self.groq_client.extract_message_text(response)
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise GroqAPIError(f"Groq returned non-JSON payload: {raw_text}") from exc

        agent_payload = payload.get("agent") or {}
        directive = GroqDirective(
            language=payload.get("language", "en-IN"),
            assistant_reply=payload.get(
                "assistant_reply",
                "मैं जानकारी एकत्र कर रही हूँ, कृपया प्रतीक्षा करें।",
            ),
            agent=AgentDirective(
                name=agent_payload.get("name", "NONE"),
                inputs=agent_payload.get("inputs", {}),
            ),
            encourage_doctor=bool(payload.get("encourage_doctor", False)),
            safety_note=payload.get(
                "safety_note",
                "If symptoms persist or worsen, please consult a qualified doctor immediately.",
            ),
        )

        if directive.agent.name != "NONE":
            wait_text = WAITING_TRANSLATIONS.get(directive.language)
            if wait_text:
                directive.assistant_reply = wait_text
        return directive

    def _run_agent(self, agent: AgentDirective) -> Optional[str]:
        if agent.name == "LOCAL_DIRECTORY":
            pincode = agent.inputs.get("pincode", "").strip()
            if not pincode:
                return None
            entries = self.local_agent.lookup(pincode)
            if not entries:
                return None
            lines = [
                f"{item['name']} | {item['address']} | {item['phone']} | {item['hours']} | {item['notes']}"
                for item in entries
            ]
            return "\n".join(lines)

        if agent.name == "HEALTH_KNOWLEDGE":
            topic = agent.inputs.get("topic", "")
            if not topic:
                return None
            return self.health_agent.fetch(topic)
        return None

    def _summarise_agent_response(
        self,
        *,
        language: str,
        agent_name: str,
        agent_inputs: Dict[str, str],
        agent_output: str,
        safety_note: str,
        encourage_doctor: bool,
    ) -> str:
        summary_prompt = AGENT_SUMMARY_PROMPT_TEMPLATE.format(language=language)
        augmented_history = self._memory_as_messages()
        agent_context = json.dumps(
            {
                "agent_name": agent_name,
                "agent_inputs": agent_inputs,
                "agent_output": agent_output,
                "safety_note": safety_note,
                "encourage_doctor": encourage_doctor,
            },
            ensure_ascii=False,
        )
        augmented_history.append(GroqMessage(role="system", content=f"AGENT_DATA::{agent_context}"))
        response = self.groq_client.structured_complete(
            summary_prompt,
            augmented_history,
        )
        return self.groq_client.extract_message_text(response)

    def _memory_as_messages(self) -> List[GroqMessage]:
        return [GroqMessage(role=turn.role, content=turn.content) for turn in self.memory.history]


__all__ = ["SakhiAssistant", "AssistantTurnResult"]
