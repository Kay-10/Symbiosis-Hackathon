# Symbiosis Hackathon – Sakhi Assistant

Sakhi is a conversational healthcare assistant designed for rural women in India. It offers a culturally-aware chat experience, supports multiple Indian languages, and can hand off to specialised agents for local doctor lookups or trusted health guidance.

## Features
- Text and voice interaction modes with persistent chat history for follow-up questions.
- Groq-powered conversational core that adapts tone and output language to the user.
- Structured agent hand-offs:
  - `LOCAL_DIRECTORY` searches a curated PIN-code directory (`sakhi_chatbot/data/local_health_directory.json`).
  - `HEALTH_KNOWLEDGE` references vetted summaries (`sakhi_chatbot/data/health_knowledge_base.json`).
- Voice pipeline reusing the existing `SpeechRecognition` flow and optional text-to-speech playback via `pyttsx3`.
- Encourages medical escalation when severe symptoms are detected and always relays a safety note.

## Setup
1. Ensure Python 3.9+ is available.
2. Install dependencies:
   ```bash
   pip install requests langdetect SpeechRecognition pyttsx3
   ```
   Voice mode additionally requires microphone support (PyAudio for SpeechRecognition).
3. Export your Groq API key:
   ```bash
   export GROQ_API_KEY="your_key_here"
   ```

## Running Sakhi
### Text mode (default)
```bash
python3 main.py --mode text
```

### Voice mode
```bash
python3 main.py --mode voice
```
Voice mode listens for speech, shows intermediate "please wait" prompts, and plays back the response using text-to-speech when available. Say “stop” or “quit” to end the session.

### Useful flags
- `--history PATH` – customise the chat history location (default `.storage/chat_history.json`).
- `--clear-history` – start a fresh conversation.
- `--model MODEL_ID` – override the Groq model (default `mixtral-8x7b-32768`).

## Customising data sources
- Update `sakhi_chatbot/data/local_health_directory.json` to enrich local doctor/clinic listings by PIN code.
- Extend `sakhi_chatbot/data/health_knowledge_base.json` with additional trusted topics. Keep entries concise and cite reliable sources.

## Notes
- The application persists history automatically so follow-up questions maintain context.
- When network access or Groq API calls fail, Sakhi returns user-friendly error messages and encourages contacting a local health worker.
