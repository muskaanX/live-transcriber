"""
Testing microphone access
"""
import sounddevice as sd
import numpy as np

# settings
DURATION = 3  # seconds
SAMPLE_RATE = 16000  # Hz (whisper expects 16kHz)

print("Microphone Test")
print("=" * 40)

# listing available audio devices
print("\nAvailable audio devices:\n")
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:  # only showing input devices
        marker = " <-- DEFAULT" if i == sd.default.device[0] else ""
        print(f"  [{i}] {device['name']}{marker}")

print(f"\nRecording {DURATION} seconds... Speak now!")

# recording audio
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype=np.float32
)
sd.wait()  # waiting for recording to complete

print("Recording complete!")

# checking if we captured any sound
volume = np.abs(audio).mean()
peak = np.abs(audio).max()

print(f"\nAudio stats:")
print(f"   Average volume: {volume:.4f}")
print(f"   Peak volume:    {peak:.4f}")

if peak > 0.01:
    print("\nMicrophone is working and captured audio.")
else:
    print("\nVery low volume detected. Check:")
    print("   - Is the correct mic selected?")
    print("   - Is the volume turned up?")
    print("   - Try speaking louder or closer to the mic.")
