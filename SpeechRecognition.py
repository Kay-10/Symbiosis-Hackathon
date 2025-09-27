import speech_recognition as sr
import time

MICROPHONE_INDEX = None
NOISE_CALIBRATION_DURATION = 1.0
RECORD_DURATION = 5
LANGUAGES = ["en-US", "hi-IN"]
STOP_WORD = "finish" 

r = sr.Recognizer()

def initialize_microphone():
    try:
        if MICROPHONE_INDEX is None:
            return sr.Microphone()
        else:
            return sr.Microphone(device_index=MICROPHONE_INDEX)
    except Exception as e:
        print(f"Error initializing microphone: {e}")
        print("Please ensure your microphone is working and PyAudio is installed.")
        return None

def filter_noise_and_calibrate(recognizer, source):
    with source:
        recognizer.energy_threshold = 3000
        recognizer.adjust_for_ambient_noise(source, duration=NOISE_CALIBRATION_DURATION)
    print("Calibration complete. System ready.")

def transcribe_audio(audio_data):
    primary_lang = LANGUAGES[0]


    try:
        text = r.recognize_google(audio_data, language=primary_lang)
        print(f"Transcription SUCCESS ({primary_lang}): \"{text}\"")
        return text

    except sr.UnknownValueError:
        fallback_lang = LANGUAGES[1]
        print(f"Transcription FAILED for {primary_lang}. Switching language to {fallback_lang}...")

        try:
            text = r.recognize_google(audio_data, language=fallback_lang)
            print(f"Transcription SUCCESS (Switched to {fallback_lang}): \"{text}\"")
            return text

        except sr.UnknownValueError:
            print("Language switch FAILED. Attempting Fallback.")
            print("Fallback: Local Whisper model would be used here.")
            return None

        except sr.RequestError as e:
            print(f"Request Error for API during transcription: {e}")
            return None

    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service: {e}")
        return None

def continuous_interaction(mic):
    # Perform calibration once at the start
    filter_noise_and_calibrate(r, mic)

    with mic as source:
        print(f"Listening continuously. Say '{STOP_WORD}' to stop.")
        all_transcriptions = []
        is_running = True

        while is_running:
            try:
                print("\n(--- READY ---) Step 1: Listening for user speech (max 5s per segment)...")

                audio = r.listen(source, timeout=None, phrase_time_limit=RECORD_DURATION)

                print("Recording segment stopped. Audio captured. Step 2: Transcribing...")

                transcribed_text = transcribe_audio(audio)

                if transcribed_text:
                    if STOP_WORD in transcribed_text.lower():
                        print(f"\nSTOP WORD ('{STOP_WORD}') DETECTED. Stopping interaction.")
                        is_running = False
                        if transcribed_text.lower().strip() != STOP_WORD:
                            all_transcriptions.append(transcribed_text)
                    else:
                        all_transcriptions.append(transcribed_text)
                        print("\nCONTINUING...")
                else:
                    print("Could not transcribe valid speech in this segment.")
                    print("CONTINUING...")

            except sr.WaitTimeoutError:
                print("No speech detected within the timeout. Continuing...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                is_running = False

        print("\n" + "="*30)
        print("FINAL ALL-SEGMENT TRANSCRIPTION:")
        print("\n".join(all_transcriptions))
        print("="*30)


if __name__ == "__main__":
    microphone = initialize_microphone()

    if microphone:
        try:
            continuous_interaction(microphone)
        except KeyboardInterrupt:
            print("\n\n--- Program Shutting Down ---")
    else:
        print("\nApplication terminated due to microphone initialization failure.")
