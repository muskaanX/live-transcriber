"""
Testing whisper transcription
"""
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# settings
DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz (whisper expects 16kHz)
MODEL_SIZE = "small"  # options: tiny, base, small, medium, large

print("Whisper Transcription Test")
print("=" * 40)

# loading whisper model (this may take a moment first time - downloads the model)
print(f"\nLoading Whisper model '{MODEL_SIZE}'...")
print("   (First run will download ~500MB model)")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded!")

# recording audio
print(f"\nRecording {DURATION} seconds... Speak now!")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype=np.float32
)
sd.wait()
print("Recording complete!")

# flattening audio array for Whisper
audio_flat = audio.flatten()

# transcribing
print("\nTranscribing...")
segments, info = model.transcribe(audio_flat, beam_size=5)

print(f"\nTranscription (detected language: {info.language}):\n")
print("-" * 40)
for segment in segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
print("-" * 40)
