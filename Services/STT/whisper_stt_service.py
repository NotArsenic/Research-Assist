import whisper
import torch
import os

STT_MODEL_SIZE = "small"

STT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- STT Service ---")
print(f"Using Whisper model: '{STT_MODEL_SIZE}'")
print(f"Using device: '{STT_DEVICE}'")

os.environ["WHISPER_NO_FP16"] = "True"

whisper_model = None

try:
    print(f"Loading Whisper model '{STT_MODEL_SIZE}' onto device '{STT_DEVICE}'...")
    
    whisper_model = whisper.load_model(STT_MODEL_SIZE, device=STT_DEVICE)
    
    print("Whisper model loaded successfully.")
    
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    
    if STT_DEVICE == "cuda":
        print("CUDA error!")

def transcribe_audio(audio_filepath: str) -> str:
    """
    Transcribes an audio file using the local Whisper model.

    Returns the final transcribed text.

    Args:
        audio_filepath: The path to the audio file.

    Returns:
        (str): The final, clean transcribed text (for the agent).
    """
    
    if whisper_model is None:
        print("Whisper model not loaded. Aborting transcription.")
        return "Error: Model not loaded."

    print(f"Transcribing {audio_filepath} with Whisper...")

    try:
        result = whisper_model.transcribe(
            audio_filepath, 
            fp16=False
        )
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error during transcription."

    final_text = result.get('text', '').strip()
    
    print(f"Transcription complete: '{final_text}'")
    
    return final_text

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_AUDIO_FILE = os.path.join(script_dir, "stt_test.wav")
    
    if os.path.exists(TEST_AUDIO_FILE) and whisper_model:
        print("\n--- Running STT Service Test ---")
        
        text = transcribe_audio(TEST_AUDIO_FILE)
        
        print("\n--- Final Text ---")
        print(text)
        print("---------------------------------")

    elif not os.path.exists(TEST_AUDIO_FILE):
        print(f"\nSkipping test: Test file not found at '{TEST_AUDIO_FILE}'.")
        
    else:
        print("\nSkipping test: Whisper model not loaded.")