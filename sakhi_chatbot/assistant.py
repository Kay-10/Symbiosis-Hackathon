"""Agent-enabled orchestration for the Sakhi healthcare assistant."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .agents.health_info_agent import HealthKnowledgeAgent
from .agents.local_info_agent import LocalDirectoryAgent
from .groq_client import GroqAPIError, GroqChatClient, GroqMessage
from .language import LanguageRouter
from .memory import ConversationMemory

PRIMARY_SYSTEM_PROMPT = """
You are Sakhi, a warm and caring community health guide supporting rural women in India.
Speak naturally and empathetically in the user's language. Keep replies conversational (3-5 sentences) and avoid clinical jargon.
Output strict JSON with this schema: {"language": string, "assistant_reply": string, "next_step": {"type": "NONE" | "LOCAL_DIRECTORY" | "HEALTH_KNOWLEDGE", "inputs": object}, "encourage_doctor": boolean}.
Guidelines:
- If symptoms are severe (heavy bleeding, high fever, severe pain, pregnancy complications, fainting, etc.), set encourage_doctor=true and clearly advise visiting a doctor. If no PIN code is known, gently ask for it instead of calling an agent.
- Only set next_step.type to LOCAL_DIRECTORY when you already have a valid 6-digit PIN code and include it in inputs.
- When the user mainly seeks self-care tips, set next_step.type to HEALTH_KNOWLEDGE with a concise lowercase topic keyword.
- Whenever next_step.type is not NONE, assistant_reply must only contain a short, friendly waiting message asking the user to hold for about a minute (no follow-up questions).
- Always follow any Preferred language instruction exactly and sound warm, respectful, and culturally sensitive. In Hindi or related languages, address the user lovingly as "बहन", "दीदी", or "सखी"; in English, use caring terms like "sister" when appropriate.
"""

AGENT_SUMMARY_PROMPT_TEMPLATE = """
You are Sakhi continuing the same conversation in {language_label} ({language}). Blend the agent data below into a caring, natural reply. Highlight key points, encourage a doctor visit when encourage_doctor is true, and stay concise.
"""

WAITING_TRANSLATIONS: Dict[str, str] = {
    "en-IN": "Please wait about a minute while I gather trusted information...",
    "hi-IN": "कृपया एक मिनट प्रतीक्षा करें, मैं भरोसेमंद जानकारी जुटा रही हूँ...",
    "bn-IN": "অনুগ্রহ করে এক মিনিট অপেক্ষা করুন, আমি নির্ভরযোগ্য তথ্য খুঁজে আনছি...",
    "te-IN": "దయచేసి ఒక నిమిషం వేచి ఉండండి, నమ్మదగిన సమాచారం తెస్తున్నాను...",
    "ta-IN": "ஒரு நிமிடம் காத்திருக்கவும், நம்பகமான தகவலை தேடிக்கொண்டு வருகிறேன்...",
    "ml-IN": "ഒരു മിനിറ്റ് കാത്തിരിക്കൂ, വിശ്വസനീയമായ വിവരങ്ങൾ ശേഖരിക്കുകയാണ്...",
    "mr-IN": "कृपया एक मिनिट थांबा, मी खात्रीशीर माहिती गोळा करत आहे...",
    "gu-IN": "મહેરબાની કરીને એક મિનિટ રાહ જુઓ, હું વિશ્વસનીય માહિતી શોધી રહી છું...",
    "kn-IN": "ದಯವಿಟ್ಟು ಒಂದು ನಿಮಿಷ ಕಾಯಿರಿ, ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿಯನ್ನು ತರ್ತಿದ್ದೇನೆ...",
    "pa-IN": "ਕਿਰਪਾ ਕਰਕੇ ਇੱਕ ਮਿੰਟ ਠਹਿਰੋ, ਮੈਂ ਭਰੋਸੇਮੰਦ ਜਾਣਕਾਰੀ ਲੈ ਰਹੀ ਹਾਂ...",
    "ur-IN": "براہ کرم ایک منٹ انتظار کریں، میں قابلِ بھروسہ معلومات لا رہی ہوں...",
}

NO_AGENT_FALLBACK: Dict[str, str] = {
    "en-IN": "I could not pull trusted details right now. Please visit or call your nearest doctor or ASHA worker as soon as possible.",
    "hi-IN": "मैं अभी भरोसेमंद जानकारी नहीं ला पा रही हूँ। कृपया जल्द से जल्द नज़दीकी डॉक्टर या आशा कार्यकर्ता से संपर्क करें।",
    "bn-IN": "এই মুহূর্তে নির্ভরযোগ্য তথ্য আনতে পারলাম না। অনুগ্রহ করে যত দ্রুত সম্ভব নিকটস্থ চিকিৎসক বা আশা কর্মীর সঙ্গে যোগাযোগ করুন।",
    "ta-IN": "நம்பகமான தகவலை இப்போது பெற முடியவில்லை. தயவு செய்து விரைவில் அருகிலுள்ள மருத்துவரையோ ஆஷா பணியாளரையோ தொடர்புக் கொள்ளுங்கள்.",
    "te-IN": "ఇప్పుడు విశ్వసనీయ సమాచారం అందలేకపోయాను. దయచేసి వెంటనే సమీప వైద్యుడిని లేదా ఆశా వర్కర్‌ని సంప్రదించండి.",
    "ml-IN": "ഇപ്പോൾ വിശ്വസനീയമായ വിവരം ലഭ്യമല്ല. ദയവായി ഉടൻ അടുത്തുള്ള ഡോക്ടറെ അല്ലെങ്കിൽ ആശാ പ്രവർത്തകയെ സമീപിക്കുക.",
    "mr-IN": "मला सध्या खात्रीशीर माहिती मिळू शकली नाही. कृपया त्वरित जवळच्या डॉक्टरांशी किंवा आशा कार्यकर्त्याशी संपर्क साधा.",
    "gu-IN": "હું હમણાં વિશ્વસનીય માહિતી મેળવી શકી નથી. કૃપા કરીને તાત્કાલિક નજીકના ડૉક્ટર અથવા આશા કાર્યકરને સંપર્ક કરો.",
    "kn-IN": "ಈ ಕ್ಷಣಕ್ಕೆ ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿಯನ್ನು ತರಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಸಾಧ್ಯವಾದಷ್ಟು ಬೇಗ ಸಮೀಪದ ವೈದ್ಯರನ್ನು ಅಥವಾ ಆಶಾ ಕಾರ್ಯಕರ್ತರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    "pa-IN": "ਮੈਨੂੰ ਇਸ ਵੇਲੇ ਭਰੋਸੇਯੋਗ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲ ਸਕੀ। ਕਿਰਪਾ ਕਰਕੇ ਜਿੰਨਾ ਜਲਦੀ ਹੋ ਸਕੇ, ਨੇੜਲੇ ਡਾਕਟਰ ਜਾਂ ਆਸ਼ਾ ਵਰਕਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
    "ur-IN": "میں اس وقت مستند معلومات نہیں لا سکی۔ براہ کرم جلد از جلد قریب ترین ڈاکٹر یا آشا ورکر سے رابطہ کریں۔",
}

TOPIC_NORMALISATION: Dict[str, str] = {
    "sir dard": "headache",
    "sar dard": "headache",
    "dard": "pain",
    "bukhar": "fever",
    "bukhaar": "fever",
    "khansi": "cough",
    "khansi bukhar": "fever",
    "pet dard": "stomach pain",
    "pet": "stomach",
    "ulati": "vomiting",
    "ulta": "vomiting",
}

LOCATION_REQUESTS: Dict[str, str] = {
    "hi-IN": "बेन, आपकी सुविधा के लिए मुझे आपका गाँव या पिन कोड बता दीजिए ताकि मैं नज़दीकी डॉक्टर खोज सकूँ।",
    "en-IN": "Didi, please share your village or PIN code so I can find a nearby doctor for you.",
}

LOCATION_REPROMPTS: Dict[str, str] = {
    "hi-IN": "कृपया छह अंकों का पिन कोड या अपने गाँव का नाम फिर से बताइए, ताकि मैं सही जगह खोज सकूँ।",
    "en-IN": "Please tell me a six digit PIN code or the name of your village again so I can look up the right place.",
}

CONFIRMATION_MESSAGES: Dict[str, str] = {
    "hi-IN": "मैंने पिन कोड {pincode} समझा है। क्या यह सही है? हाँ या नहीं में बताइए।",
    "en-IN": "I understood the PIN code as {pincode}. Is that correct? Please reply with yes or no.",
}

CONFIRMATION_REPROMPTS: Dict[str, str] = {
    "hi-IN": "कृपया हाँ या नहीं में बताइए ताकि मैं आपकी मदद जारी रख सकूँ।",
    "en-IN": "Please answer with yes or no so I can keep helping you.",
}

AFFIRMATIVE_RESPONSES = {
    "haan",
    "ha",
    "haanji",
    "han",
    "yes",
    "h",
    "bilkul",
    "ji",
    "sahi",
    "correct",
    "y",
}

NEGATIVE_RESPONSES = {
    "nahin",
    "nahi",
    "no",
    "n",
    "galat",
    "wrong",
    "na",
    "nopes",
}

PIN_EXTRACTION_PROMPT = """
You are an assistant who extracts Indian postal PIN codes (six digit numbers) from short location descriptions.
Return JSON with keys `pincode` (string, six digits or empty), `confidence` (float between 0 and 1) and `reason` (string).
If you cannot determine a PIN code, set `pincode` to an empty string.
Do not include any additional text outside JSON.
"""

LANGUAGE_LABELS: Dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "pa-IN": "Punjabi",
    "ur-IN": "Urdu",
}

SUMMARY_HINTS: Dict[str, str] = {
    "en-IN": "Please share the information below with the sister in caring English, keeping a warm, supportive tone.",
    "hi-IN": "कृपया नीचे की जानकारी बहन के साथ सहज और देखभाल भरे तरीके से हिंदी में साझा करें।",
    "bn-IN": "অনুগ্রহ করে নিচের তথ্যটি বোনের সঙ্গে যত্নশীল ভঙ্গিতে বাংলায় ভাগ করুন।",
    "te-IN": "దయచేసి క్రింది సమాచారాన్ని సోదరితో ఆప్యాయంగా తెలుగులో పంచుకోండి.",
    "ta-IN": "தயவுசெய்து கீழுள்ள தகவலை அக்காவுடன் அன்பான தமிழில் பகிரவும்.",
    "ml-IN": "ദയവായി താഴെയുള്ള വിവരം സഹോദരിയോട് കരുതലോടെ മലയാളത്തിൽ പങ്കിടുക.",
    "mr-IN": "कृपया खालील माहिती बहीणीसोबत प्रेमळ मराठीत सांगा.",
    "gu-IN": "કૃપા કરીને નીચેની માહિતી બહેન સાથે પ્રેમથી ગુજરાતીમાં વહેંચો.",
    "kn-IN": "ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಮಾಹಿತಿಯನ್ನು ಸಹೋದರಿಯ ಜೊತೆ ಮಮತೆಯಿಂದ ಕನ್ನಡದಲ್ಲಿ ಹಂಚಿಕೊಳ್ಳಿ.",
    "pa-IN": "ਕਿਰਪਾ ਕਰਕੇ ਹੇਠਾਂ ਦਿੱਤੀ ਜਾਣਕਾਰੀ ਬਹਿਨ ਨਾਲ ਪਿਆਰ ਨਾਲ ਪੰਜਾਬੀ ਵਿੱਚ ਸਾਂਝੀ ਕਰੋ।",
    "ur-IN": "براہ کرم نیچے دی گئی معلومات بہن کے ساتھ پیار سے اردو میں شیئر کریں۔",
}


@dataclass
class AgentDirective:
    type: str = "NONE"
    inputs: Dict[str, str] = field(default_factory=dict)


@dataclass
class GroqDirective:
    language: str
    assistant_reply: str
    next_step: AgentDirective
    encourage_doctor: bool


@dataclass
class AssistantTurnResult:
    language: str
    message: str
    encourage_doctor: bool
    agent_name: str
    agent_inputs: Dict[str, str]
    agent_output: Optional[str] = None
    intermediate_message: Optional[str] = None


class SakhiAssistant:
    """Coordinate Groq reasoning with optional agent hand-offs."""

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
        data_root = Path(__file__).resolve().parent / "data"
        self.local_agent = local_agent or LocalDirectoryAgent(
            data_root / "local_health_directory.json"
        )
        self.health_agent = health_agent or HealthKnowledgeAgent(
            data_root / "health_knowledge_base.json"
        )
        self.known_pincode: Optional[str] = None
        self.awaiting_location: bool = False
        self.pending_pincode: Optional[str] = None
        self.awaiting_pincode_confirmation: bool = False

    def handle_user_message(
        self,
        user_text: str,
        *,
        on_intermediate: Optional[Callable[[str, str], None]] = None,
    ) -> AssistantTurnResult:
        lang_decision = self.language_router.detect_language(user_text)
        language_code = lang_decision.language_code
        self.memory.append("user", user_text)

        if self.awaiting_pincode_confirmation and self.pending_pincode:
            return self._handle_pincode_confirmation(
                user_text,
                language_code,
                on_intermediate,
            )

        if self.awaiting_location:
            return self._handle_location_response(user_text, language_code)

        self._capture_inline_pincode(user_text)

        conversation = self._memory_as_messages()
        directive = self._query_groq(
            conversation,
            language_hint=language_code,
        )
        language_code = directive.language

        self._capture_inline_pincode(directive.assistant_reply)

        if directive.encourage_doctor and not self._latest_pincode():
            request = self._location_request(language_code)
            self.awaiting_location = True
            self.pending_pincode = None
            self.awaiting_pincode_confirmation = False
            self.memory.append("assistant", request)
            return AssistantTurnResult(
                language=language_code,
                message=request,
                encourage_doctor=True,
                agent_name="NONE",
                agent_inputs={},
                intermediate_message=None,
            )

        self.memory.append("assistant", directive.assistant_reply)
        if directive.next_step.type == "NONE":
            return AssistantTurnResult(
                language=language_code,
                message=directive.assistant_reply,
                encourage_doctor=directive.encourage_doctor,
                agent_name="NONE",
                agent_inputs={},
                intermediate_message=None,
            )

        if (
            directive.next_step.type == "LOCAL_DIRECTORY"
            and self._is_valid_pincode(directive.next_step.inputs.get("pincode", ""))
        ):
            self.known_pincode = directive.next_step.inputs.get("pincode")

        if on_intermediate:
            on_intermediate(directive.assistant_reply, language_code)

        agent_output = self._run_agent(directive.next_step)

        if directive.next_step.type == "HEALTH_KNOWLEDGE" and not agent_output:
            latest_pin = self._latest_pincode()
            if latest_pin:
                self.known_pincode = latest_pin
                return self._respond_with_local_directory(
                    language_code,
                    on_intermediate,
                    encourage=True,
                    emit_wait=False,
                )
            self.awaiting_location = True
            self.awaiting_pincode_confirmation = False
            self.pending_pincode = None
            request = self._location_request(language_code)
            self.memory.append("assistant", request)
            return AssistantTurnResult(
                language=language_code,
                message=request,
                encourage_doctor=True,
                agent_name="NONE",
                agent_inputs={},
                intermediate_message=None,
            )

        if not agent_output:
            agent_output = NO_AGENT_FALLBACK.get(
                language_code, NO_AGENT_FALLBACK["en-IN"]
            )

        agent_summary = self._summarise_agent_response(
            language=language_code,
            agent_name=directive.next_step.type,
            agent_inputs=directive.next_step.inputs,
            agent_output=agent_output,
            encourage_doctor=directive.encourage_doctor,
        )

        self.memory.append("assistant", agent_summary)
        return AssistantTurnResult(
            language=language_code,
            message=agent_summary,
            encourage_doctor=directive.encourage_doctor,
            agent_name=directive.next_step.type,
            agent_inputs=directive.next_step.inputs,
            agent_output=agent_output,
            intermediate_message=directive.assistant_reply,
        )

    def _respond_with_local_directory(
        self,
        language: str,
        on_intermediate: Optional[Callable[[str, str], None]],
        *,
        encourage: bool,
        emit_wait: bool,
    ) -> AssistantTurnResult:
        pincode = self._latest_pincode()
        wait_text = WAITING_TRANSLATIONS.get(language, WAITING_TRANSLATIONS["en-IN"])
        self.awaiting_location = False
        self.awaiting_pincode_confirmation = False
        self.pending_pincode = None
        intermediate_message = wait_text if emit_wait else None
        if emit_wait:
            self.memory.append("assistant", wait_text)
            if on_intermediate:
                on_intermediate(wait_text, language)
        directive = AgentDirective(
            type="LOCAL_DIRECTORY",
            inputs={"pincode": pincode} if pincode else {},
        )
        agent_output = self._run_agent(directive)
        if not agent_output:
            agent_output = NO_AGENT_FALLBACK.get(language, NO_AGENT_FALLBACK["en-IN"])
        agent_summary = self._summarise_agent_response(
            language=language,
            agent_name="LOCAL_DIRECTORY",
            agent_inputs=directive.inputs,
            agent_output=agent_output,
            encourage_doctor=encourage,
        )
        self.memory.append("assistant", agent_summary)
        return AssistantTurnResult(
            language=language,
            message=agent_summary,
            encourage_doctor=encourage,
            agent_name="LOCAL_DIRECTORY",
            agent_inputs=directive.inputs,
            agent_output=agent_output,
            intermediate_message=intermediate_message,
        )

    def _handle_location_response(
        self,
        user_text: str,
        language: str,
    ) -> AssistantTurnResult:
        pincode = self._extract_pincode_from_text(user_text)
        if pincode:
            self.pending_pincode = pincode
            self.awaiting_location = False
            self.awaiting_pincode_confirmation = True
            confirmation = self._confirmation_prompt(language, pincode)
            self.memory.append("assistant", confirmation)
            return AssistantTurnResult(
                language=language,
                message=confirmation,
                encourage_doctor=True,
                agent_name="NONE",
                agent_inputs={},
                intermediate_message=None,
            )
        reprompt = self._location_reprompt(language)
        self.memory.append("assistant", reprompt)
        return AssistantTurnResult(
            language=language,
            message=reprompt,
            encourage_doctor=True,
            agent_name="NONE",
            agent_inputs={},
            intermediate_message=None,
        )

    def _handle_pincode_confirmation(
        self,
        user_text: str,
        language: str,
        on_intermediate: Optional[Callable[[str, str], None]],
    ) -> AssistantTurnResult:
        if self._is_affirmative(user_text):
            known = self.pending_pincode or ""
            if self._is_valid_pincode(known):
                self.known_pincode = known
            self.pending_pincode = None
            self.awaiting_pincode_confirmation = False
            self.awaiting_location = False
            return self._respond_with_local_directory(
                language,
                on_intermediate,
                encourage=True,
                emit_wait=True,
            )

        if self._is_negative(user_text):
            self.pending_pincode = None
            self.awaiting_pincode_confirmation = False
            self.awaiting_location = True
            reprompt = self._location_reprompt(language)
            self.memory.append("assistant", reprompt)
            return AssistantTurnResult(
                language=language,
                message=reprompt,
                encourage_doctor=True,
                agent_name="NONE",
                agent_inputs={},
                intermediate_message=None,
            )

        reminder = self._confirmation_retry(language)
        self.memory.append("assistant", reminder)
        return AssistantTurnResult(
            language=language,
            message=reminder,
            encourage_doctor=True,
            agent_name="NONE",
            agent_inputs={},
            intermediate_message=None,
        )

    def _capture_inline_pincode(self, text: str) -> None:
        if self.known_pincode:
            return
        candidate = self._regex_pincode(text)
        if candidate:
            self.known_pincode = candidate
            self.pending_pincode = None
            self.awaiting_location = False
            self.awaiting_pincode_confirmation = False

    @staticmethod
    def _regex_pincode(text: str) -> Optional[str]:
        match = re.search(r"[1-9][0-9]{5}", text)
        if match:
            return match.group(0)
        return None

    def _extract_pincode_from_text(self, text: str) -> Optional[str]:
        candidate = self._regex_pincode(text)
        if candidate:
            return candidate
        return self._extract_pincode_via_groq(text)

    def _extract_pincode_via_groq(self, text: str) -> Optional[str]:
        if not text.strip():
            return None
        messages = [
            GroqMessage(role="system", content=PIN_EXTRACTION_PROMPT),
            GroqMessage(role="user", content=text.strip()),
        ]
        try:
            response = self.groq_client.complete(
                messages,
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
        except GroqAPIError:
            return None
        except Exception:
            return None
        raw = self.groq_client.extract_message_text(response)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        candidate = str(data.get("pincode", "")).strip()
        if self._is_valid_pincode(candidate):
            return candidate
        return None

    def _location_request(self, language: str) -> str:
        return LOCATION_REQUESTS.get(language, LOCATION_REQUESTS["en-IN"])

    def _location_reprompt(self, language: str) -> str:
        return LOCATION_REPROMPTS.get(language, LOCATION_REPROMPTS["en-IN"])

    def _confirmation_prompt(self, language: str, pincode: str) -> str:
        template = CONFIRMATION_MESSAGES.get(language, CONFIRMATION_MESSAGES["en-IN"])
        return template.format(pincode=pincode)

    def _confirmation_retry(self, language: str) -> str:
        return CONFIRMATION_REPROMPTS.get(language, CONFIRMATION_REPROMPTS["en-IN"])

    def _is_affirmative(self, text: str) -> bool:
        tokens = text.lower().split()
        if text.strip().lower() in AFFIRMATIVE_RESPONSES:
            return True
        return any(token in AFFIRMATIVE_RESPONSES for token in tokens)

    def _is_negative(self, text: str) -> bool:
        tokens = text.lower().split()
        if text.strip().lower() in NEGATIVE_RESPONSES:
            return True
        return any(token in NEGATIVE_RESPONSES for token in tokens)

    def _query_groq(
        self,
        conversation: List[GroqMessage],
        *,
        language_hint: Optional[str] = None,
    ) -> GroqDirective:
        messages = self._build_prompt_messages(conversation, language_hint)
        payload = self._complete_with_json(messages)

        next_payload = payload.get("next_step") or {}
        raw_inputs = next_payload.get("inputs", {})
        inputs = raw_inputs if isinstance(raw_inputs, dict) else {}

        agent_type = next_payload.get("type", "NONE")
        if agent_type == "LOCAL_DIRECTORY":
            pincode = inputs.get("pincode", "").strip()
            if not self._is_valid_pincode(pincode):
                agent_type = "NONE"
                inputs = {}
        if agent_type == "HEALTH_KNOWLEDGE" and not inputs.get("topic"):
            inputs["topic"] = self._fallback_topic()
        if agent_type == "HEALTH_KNOWLEDGE":
            inputs["topic"] = self._normalise_topic(inputs.get("topic", ""))

        assistant_text = payload.get("assistant_reply", "")
        encourage_flag = bool(payload.get("encourage_doctor", False))
        if not encourage_flag:
            lowered_text = assistant_text.lower()
            doctor_terms = ["doctor", "clinic", "hospital", "डॉक्टर", "अस्पताल", "डाक्टर"]
            if any(term in lowered_text for term in doctor_terms):
                encourage_flag = True
        latest_pin = self._latest_pincode()
        if encourage_flag and agent_type != "LOCAL_DIRECTORY" and latest_pin:
            agent_type = "LOCAL_DIRECTORY"
            inputs = {"pincode": latest_pin}

        language_selected = language_hint or payload.get("language", "en-IN")
        directive = GroqDirective(
            language=language_selected,
            assistant_reply=assistant_text or WAITING_TRANSLATIONS.get(
                language_selected, WAITING_TRANSLATIONS["en-IN"]
            ),
            next_step=AgentDirective(type=agent_type, inputs=inputs),
            encourage_doctor=encourage_flag,
        )

        if directive.next_step.type != "NONE":
            wait_text = WAITING_TRANSLATIONS.get(
                directive.language, WAITING_TRANSLATIONS["en-IN"]
            )
            directive.assistant_reply = wait_text
        return directive

    def _build_prompt_messages(
        self,
        conversation: List[GroqMessage],
        language_hint: Optional[str],
    ) -> List[GroqMessage]:
        messages: List[GroqMessage] = [GroqMessage(role="system", content=PRIMARY_SYSTEM_PROMPT)]
        context_hint = self._context_hint()
        if context_hint:
            messages.append(
                GroqMessage(role="system", content=f"Known context: {context_hint}")
            )
        if language_hint:
            label = self._language_label(language_hint)
            messages.append(
                GroqMessage(
                    role="system",
                    content=(
                        f"Preferred language: {language_hint}. Respond only in {label} using a warm, caring tone."
                    ),
                )
            )
        messages.extend(conversation)
        return messages

    def _complete_with_json(self, messages: List[GroqMessage]) -> Dict[str, object]:
        reminder = GroqMessage(
            role="system",
            content=(
                "Reminder: respond ONLY with valid JSON matching the specified schema. Do not add explanations."
            ),
        )
        working_messages = list(messages)
        last_error: Optional[GroqAPIError] = None
        for _ in range(3):
            response = self.groq_client.complete(
                working_messages,
                temperature=0.3,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            raw_text = self.groq_client.extract_message_text(response)
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                last_error = GroqAPIError(
                    f"Groq returned non-JSON payload: {raw_text}"
                )
                working_messages.insert(1, reminder)
        if last_error:
            raise last_error
        raise GroqAPIError("Groq did not provide valid JSON response")

    def _run_agent(self, directive: AgentDirective) -> Optional[str]:
        if directive.type == "LOCAL_DIRECTORY":
            pincode = directive.inputs.get("pincode", "").strip()
            if not self._is_valid_pincode(pincode):
                return None
            return self.local_agent.formatted_directory(pincode)
        if directive.type == "HEALTH_KNOWLEDGE":
            topic = directive.inputs.get("topic", "").strip()
            if not topic:
                topic = self._fallback_topic()
            return self.health_agent.fetch(topic)
        return None

    def _summarise_agent_response(
        self,
        *,
        language: str,
        agent_name: str,
        agent_inputs: Dict[str, str],
        agent_output: str,
        encourage_doctor: bool,
    ) -> str:
        label = self._language_label(language)
        summary_prompt = AGENT_SUMMARY_PROMPT_TEMPLATE.format(
            language=language,
            language_label=label,
        )
        history = self._memory_as_messages()
        history.append(
            GroqMessage(
                role="system",
                content=(
                    f"Preferred language: {language}. Respond only in {label} with a warm, supportive tone."
                ),
            )
        )
        agent_context = json.dumps(
            {
                "agent_name": agent_name,
                "agent_inputs": agent_inputs,
                "agent_output": agent_output,
                "encourage_doctor": encourage_doctor,
            },
            ensure_ascii=False,
        )
        summary_hint = SUMMARY_HINTS.get(language, SUMMARY_HINTS["en-IN"])
        history.append(
            GroqMessage(
                role="user",
                content=f"AGENT_DATA::{agent_context}\n{summary_hint}",
            )
        )
        response = self.groq_client.complete(
            history,
            temperature=0.35,
            max_tokens=500,
        )
        return self.groq_client.extract_message_text(response)

    def _memory_as_messages(self) -> List[GroqMessage]:
        return [GroqMessage(role=turn.role, content=turn.content) for turn in self.memory.history]

    def _context_hint(self) -> Optional[str]:
        pincode = self._latest_pincode()
        if not pincode:
            return None
        return json.dumps({"known_pincode": pincode})

    def _latest_pincode(self) -> Optional[str]:
        if self.known_pincode and self._is_valid_pincode(self.known_pincode):
            return self.known_pincode
        pattern = re.compile(r"\b[1-9][0-9]{5}\b")
        for turn in reversed(self.memory.history):
            match = pattern.search(turn.content)
            if match:
                return match.group(0)
        return None

    def _fallback_topic(self) -> str:
        ignore = {"nahi", "nahin", "no", "haan", "yes", "the", "aur"}
        pattern = re.compile(r"[A-Za-z]+")
        for turn in reversed(self.memory.history):
            if turn.role != "user":
                continue
            tokens = [match.group(0).lower() for match in pattern.finditer(turn.content)]
            for token in tokens:
                if token not in ignore:
                    return token
        return "health"

    @staticmethod
    def _is_valid_pincode(value: str) -> bool:
        return bool(re.fullmatch(r"[1-9][0-9]{5}", value or ""))

    @staticmethod
    def _language_label(language_code: str) -> str:
        return LANGUAGE_LABELS.get(language_code, language_code)

    @staticmethod
    def _normalise_topic(topic: str) -> str:
        cleaned = topic.strip().lower()
        if not cleaned:
            return "health"
        return TOPIC_NORMALISATION.get(cleaned, cleaned)


__all__ = ["SakhiAssistant", "AssistantTurnResult"]
