import speech_recognition as sr
import time

MICROPHONE_INDEX = None 
NOISE_CALIBRATION_DURATION = 1.0 
RECORD_DURATION = 5 
LANGUAGES = ["en-US", "hi-IN"] 

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
        recognizer.energy_threshold = 4000 
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

def one_shot_interaction(mic): 
    filter_noise_and_calibrate(r, mic)
    
    
    with mic as source:
        try:
            print("\n(--- READY ---) Step 1: Listening for user speech...")
            
            audio = r.listen(source, timeout=None, phrase_time_limit=RECORD_DURATION)
            
            print("Recording stopped. Audio captured.")
            
            transcribed_text = transcribe_audio(audio)

            if not transcribed_text:
                print("Could not transcribe valid speech. Interaction ended.")
                return 
            
            print("\nFINAL TRANSCRIPTION:")
            print(transcribed_text)
            
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout. Interaction ended.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            

if __name__ == "__main__":
    microphone = initialize_microphone()
    
    if microphone:
        try:
            one_shot_interaction(microphone)
        except KeyboardInterrupt:
            print("\n\n--- Program Shutting Down ---")
    else:
        print("\nApplication terminated due to microphone initialization failure.")