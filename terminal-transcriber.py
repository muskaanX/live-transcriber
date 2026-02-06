#!/usr/bin/env python3
"""
Live Transcriber - Real-time speech transcription with web viewer
Uses Whisper for transcription. Open http://localhost:5000 to view live.
"""
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import json
import os
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from faster_whisper import WhisperModel

# ============================================
# CONFIGURATION
# ============================================
SAMPLE_RATE = 16000          # Hz (Whisper expects 16kHz)
MODEL_SIZE = "small"         # tiny, base, small, medium, large
SILENCE_THRESHOLD = 0.02     # Volume below this = silence
SILENCE_DURATION = 1.5       # Seconds of silence to trigger transcription
MIN_AUDIO_LENGTH = 0.3       # Minimum seconds of audio to transcribe
CHUNK_SIZE = 1024            # Audio buffer size
WEB_PORT = 5000              # Port for web viewer

# ============================================
# GLOBAL STATE
# ============================================
audio_queue = queue.Queue()
is_recording = True
transcript = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_transcript_json():
    """Save transcript to JSON for web viewer."""
    json_path = os.path.join(SCRIPT_DIR, "transcript.json")
    with open(json_path, "w") as f:
        json.dump(transcript, f)

def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk."""
    if status:
        print(f"⚠️  Audio status: {status}")
    audio_queue.put(indata.copy())

def get_volume(audio):
    """Calculate RMS volume of audio."""
    return np.sqrt(np.mean(audio**2))

def transcription_worker(model):
    """Background worker that processes audio and transcribes."""
    global is_recording, transcript
    
    audio_buffer = []
    silence_start = None
    last_vol_print = 0
    
    while is_recording:
        try:
            chunk = audio_queue.get(timeout=0.1)
            audio_buffer.append(chunk)
            
            volume = get_volume(chunk)
            
            # Show volume meter every 0.3 sec
            if time.time() - last_vol_print > 0.3:
                bar = "█" * int(min(volume * 500, 40)) + "░" * (40 - int(min(volume * 500, 40)))
                status = "🔊" if volume >= SILENCE_THRESHOLD else "🔇"
                print(f"\r{status} [{bar}] {volume:.4f}  ", end="", flush=True)
                last_vol_print = time.time()
            
            if volume < SILENCE_THRESHOLD:
                # Silence detected
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    # Enough silence - transcribe what we have
                    if len(audio_buffer) > 0:
                        audio_data = np.concatenate(audio_buffer).flatten()
                        duration = len(audio_data) / SAMPLE_RATE
                        
                        if duration >= MIN_AUDIO_LENGTH:
                            print(f"\r⏳ Transcribing {duration:.1f}s of audio...                    ")
                            
                            try:
                                segments, info = model.transcribe(
                                    audio_data, 
                                    beam_size=5,
                                    language="en"
                                )
                                text = " ".join([s.text for s in segments]).strip()
                                
                                if text and not text.isspace():
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    transcript.append({"time": timestamp, "text": text})
                                    save_transcript_json()  # Update JSON for web viewer
                                    print(f"\r[{timestamp}] {text}")
                                    print()
                                else:
                                    print(f"\r(no speech detected)                    ")
                            except Exception as e:
                                print(f"\r⚠️  Transcription error: {e}")
                    
                    # Reset buffer
                    audio_buffer = []
                    silence_start = None
            else:
                # Sound detected - reset silence timer
                silence_start = None
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️  Error: {e}")

def save_transcript_md():
    """Save transcript to markdown file."""
    if not transcript:
        print("⚠️  No transcript to save!")
        return
    
    filename = os.path.join(SCRIPT_DIR, f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(filename, "w") as f:
        f.write("# Transcript\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        for entry in transcript:
            f.write(f"**[{entry['time']}]** {entry['text']}\n\n")
    
    print(f"\n💾 Saved to: {filename}")

class QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves files from script directory."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs
    
    def do_GET(self):
        # Serve viewer.html as index
        if self.path == "/" or self.path == "/index.html":
            self.path = "/viewer.html"
        super().do_GET()

def start_web_server():
    """Start the web server in background."""
    server = HTTPServer(("", WEB_PORT), QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    return server

def main():
    global is_recording
    
    print("=" * 50)
    print("🎙️  LIVE TRANSCRIBER")
    print("=" * 50)
    
    # Initialize empty transcript JSON
    save_transcript_json()
    
    # Start web server
    server = start_web_server()
    print(f"\n🌐 Web viewer: http://localhost:{WEB_PORT}")
    
    # Load model
    print(f"\n⏳ Loading Whisper model '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("✅ Model loaded!")
    
    print("\n" + "=" * 50)
    print("Just speak! Text appears after short pauses.")
    print("Open the web viewer to see transcriptions live.")
    print("Press Ctrl+C or type 'q' + Enter to quit.")
    print("=" * 50)
    print("\n🔴 LISTENING...\n")
    
    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    
    # Start transcription worker
    worker = threading.Thread(target=transcription_worker, args=(model,))
    worker.daemon = True
    
    try:
        stream.start()
        worker.start()
        
        while True:
            cmd = input().strip().lower()
            if cmd == 'q':
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n⏹️  Stopping...")
        is_recording = False
        stream.stop()
        stream.close()
        
        save_transcript_md()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
