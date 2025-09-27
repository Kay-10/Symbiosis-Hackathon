# Symbiosis Hackathon – Sakhi Assistant

Sakhi is a conversational healthcare companion crafted for rural women in India. She listens with empathy, understands multiple Indian languages, and can escalate to trusted agents for local doctor lookups or reliable health facts whenever required.

## Features
- Warm, multilingual chat experience powered by Groq with memory-aware follow ups.
- Push-to-talk web companion with automatic silence detection, Gemini-powered transcription, and natural voice replies.
- Seamless agent hand-offs:
  - `LOCAL_DIRECTORY` fetches nearby clinics from `sakhi_chatbot/data/local_health_directory.json`.
  - `HEALTH_KNOWLEDGE` blends cached guidance with live data (US HHS MyHealthfinder, National Health Portal of India, UNICEF India, optional custom API).
- Critical cases trigger a location-to-PIN workflow so Sakhi can surface actionable doctor details before signing off.
- Text-only CLI retained for minimal environments.

## Setup
1. Ensure Python 3.9+ is installed.
2. Install dependencies from the bundled requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. Export your Groq API key:
   ```bash
   export GROQ_API_KEY="your_key_here"
   ```
4. Export your Gemini API key (used for speech-to-text):
   ```bash
   export GEMINI_API_KEY="your_gemini_key_here"
   ```
   You can override the default model by setting `GEMINI_STT_MODEL` (defaults to `gemini-1.5-flash`).
5. *(Optional)* If you have a trusted public health information API, expose it via:
   ```bash
   export HEALTH_INFO_API="https://example.org/health"
   ```
   Sakhi will call this endpoint with `?topic=<keyword>` and merge the response.

## Running Sakhi
### Text CLI (legacy)
```bash
python3 main.py
```

### Web companion (recommended)
```bash
uvicorn sakhi_chatbot.web_app:app --reload
```
Open http://localhost:8000 and use the **Talk** button for voice conversations or type messages directly. Spoken turns and typed turns share the same chat history, and responses are read aloud in the detected language. While Sakhi is processing or speaking, the Talk button is temporarily disabled; you can always press **Stop** to cancel and start a fresh recording. The web companion streams natural Microsoft neural voices (via `edge-tts`) and automatically picks Hindi, Marathi, or English based on what you say.

## Customising data sources
- Enrich `sakhi_chatbot/data/local_health_directory.json` with more clinics keyed by PIN code.
- Extend `sakhi_chatbot/data/health_knowledge_base.json` with concise, cited topics for local conditions.

## Notes
- Each web session maintains its own persisted history under `.storage/web_sessions/` so conversations can resume after refresh.
- When Gemini cannot decode audio, Sakhi prompts the user to try again without losing context.
- If Groq or network calls fail, Sakhi responds gracefully and encourages contacting a nearby health worker.
