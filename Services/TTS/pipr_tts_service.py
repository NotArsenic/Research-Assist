import os
import torch
import wave
import time
from piper.voice import PiperVoice

SCRIPT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(SCRIPT_PATH)
SERVICES_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(SERVICES_DIR)

TTS_MODELS_DIR = os.path.join(BASE_DIR, "models", "TTS")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- TTS Service ---")

print(f"Device detected: '{device}'")
print(f"Looking for models in: {TTS_MODELS_DIR}")

VOICE_MODEL_MAP = {
    "joe": "joe/en_US-joe-medium.onnx",
    "kathleen": "kathleen/en_US-kathleen-low.onnx",
    "l2arctic_male1": "l2arctic_male1/en_US-l2arctic-medium.onnx",
    "kristin": "kristin/en_US-kristin-medium.onnx",
}

LOADED_VOICES = {}

print("Loading TTS voices...")

for voice_name, model_path in VOICE_MODEL_MAP.items():

    onnx_path = os.path.join(TTS_MODELS_DIR, model_path)
    json_path = f"{onnx_path}.json"

    if not os.path.exists(onnx_path):
        print(f"  [WARNING] Voice '{voice_name}': Model file not found at {onnx_path}")
        continue

    if not os.path.exists(json_path):
        print(f"  [WARNING] Voice '{voice_name}': Config file not found at {json_path}")
        continue

    try:
        voice = PiperVoice.load(onnx_path, config_path=json_path)
        LOADED_VOICES[voice_name] = voice

        print(f"  [SUCCESS] Loaded voice: '{voice_name}'")

    except Exception as e:
        print(f"  [ERROR] Failed to load voice '{voice_name}': {e}")

if not LOADED_VOICES:
    print("[CRITICAL ERROR] No TTS voices were successfully loaded. TTS will not work.")

else:
    print(f"\nSuccessfully loaded {len(LOADED_VOICES)} TTS voice(s).")


def generate_audio(text: str, voice_name: str, output_wav_path: str) -> str | None:
    """
    Generates speech from text using a pre-loaded Piper voice.

    Args:
        text: The text to synthesize.
        voice_name: Name of the voice (as defined in VOICE_MODEL_MAP).
        output_wav_path: The file path to save the generated .wav file.

    Returns:
        The output_wav_path if successful, None otherwise.
    """

    if not LOADED_VOICES:
        print("Error: No TTS models are loaded.")
        return None

    if voice_name not in LOADED_VOICES:
        print(
            f"Error: Voice '{voice_name}' is not loaded. Available voices: {list(LOADED_VOICES.keys())}"
        )

        voice_name = list(LOADED_VOICES.keys())[0]

        print(f"Falling back to first available voice: '{voice_name}'")

    voice = LOADED_VOICES[voice_name]

    try:
        print(f"\nGenerating audio for voice: '{voice_name}'...\n")

        with wave.open(output_wav_path, "wb") as wav_file:

            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)

            for audio_chunk in voice.synthesize(text):
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        print(f"Successfully saved audio to {output_wav_path}")
        return output_wav_path

    except Exception as e:
        print(f"Error during audio synthesis: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":

    if LOADED_VOICES:
        print("\n--- Running TTS Service Test ---")

        test_voice = list(LOADED_VOICES.keys())[3]
        TEST_TEXT = "The way of life can be free and beautiful, but we have lost the way. Greed has poisoned men's souls, has barricaded the world with hate, has goose-stepped us into misery and bloodshed... We are all too clever. We are all too clever, but we are not kind enough. More than machinery we need humanity. More than cleverness we need kindness and gentleness... charlie chaplin"

        TEST_OUTPUT_FILE = os.path.join(CURRENT_DIR, "tts_test_output.wav")

        print(f"Using voice: '{test_voice}'")
        print(f"Synthesizing text: '{TEST_TEXT}'")

        start_time = time.time()
        output_path = generate_audio(TEST_TEXT, test_voice, TEST_OUTPUT_FILE)
        end_time = time.time()

        if output_path:
            print(f"Test complete in {end_time - start_time:.2f} seconds.")
            print(f"Check for the output file: {output_path}")

        else:
            print("Test failed. See errors above.")

        print("---------------------------------")

    else:
        print(
            "\nSkipping test: No TTS models were loaded. Check paths and model files."
        )
