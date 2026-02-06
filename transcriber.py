"""
Open http://localhost:5000 to view live.

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


# CONFIGURATION
SAMPLE_RATE = 16000          # Hz (Whisper expects 16kHz)
MODEL_SIZE = "small"         # tiny, base, small, medium, large
SILENCE_THRESHOLD = 0.015    # volume below this = silence
SILENCE_DURATION = 1.5       # seconds of silence to trigger transcription
MIN_AUDIO_LENGTH = 0.3       # minimum seconds of audio to transcribe
CHUNK_SIZE = 1024            # audio buffer size
WEB_PORT = 8080              # port for web viewer


# GLOBAL STATE
audio_queue = queue.Queue()
is_recording = True
transcript = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_transcript_json():
    # saving transcript to JSON for web viewer
    json_path = os.path.join(SCRIPT_DIR, "transcript.json")
    with open(json_path, "w") as f:
        json.dump(transcript, f)

def audio_callback(indata, frames, time_info, status):
    # called by sounddevice for each audio chunk
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def get_volume(audio):
    # calculating RMS volume of audio
    return np.sqrt(np.mean(audio**2))

def transcription_worker(model):
    # background worker that processes audio and transcribes
    global is_recording, transcript
    
    audio_buffer = []
    silence_start = None
    last_vol_print = 0
    heard_speech = False  # tracking if we actually heard speech
    
    while is_recording:
        try:
            chunk = audio_queue.get(timeout=0.1)
            volume = get_volume(chunk)
            
            # showing volume meter every 0.3 sec
            if time.time() - last_vol_print > 0.3:
                bar = "█" * int(min(volume * 500, 40)) + "░" * (40 - int(min(volume * 500, 40)))
                status = "🔊" if volume >= SILENCE_THRESHOLD else "🔇"
                print(f"\r{status} [{bar}] {volume:.4f}  ", end="", flush=True)
                last_vol_print = time.time()
            
            if volume >= SILENCE_THRESHOLD:
                # sound detected - buffer it and reset silence timer
                audio_buffer.append(chunk)
                heard_speech = True
                silence_start = None
            else:
                # silence detected
                if heard_speech:
                    # buffer during silence after speech
                    audio_buffer.append(chunk)
                    
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        # sufficient silence after speech - transcribe
                        if len(audio_buffer) > 0:
                            audio_data = np.concatenate(audio_buffer).flatten()
                            duration = len(audio_data) / SAMPLE_RATE
                            
                            if duration >= MIN_AUDIO_LENGTH:
                                print(f"\rTranscribing {duration:.1f}s of audio...                    ")
                                
                                try:
                                    segments, info = model.transcribe(
                                        audio_data, 
                                        beam_size=5,
                                        language="en"
                                    )
                                    text = " ".join([s.text for s in segments]).strip()
                                    
                                    if text and not text.isspace() and text.lower() not in ["you", "you.", "the", "a", "."]:
                                        timestamp = datetime.now().strftime("%H:%M:%S")
                                        transcript.append({"time": timestamp, "text": text})
                                        save_transcript_json()
                                        print(f"\r[{timestamp}] {text}")
                                        print()
                                    else:
                                        print(f"\r(no speech detected)                    ")
                                except Exception as e:
                                    print(f"\rTranscription error: {e}")
                        
                        # resetting for next speech
                        audio_buffer = []
                        silence_start = None
                        heard_speech = False
                # discarding silence if no speech heard yet
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error: {e}")

def save_transcript_md():
    # saving transcript to markdown file
    if not transcript:
        print("No transcript to save!")
        return
    
    filename = os.path.join(SCRIPT_DIR, f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(filename, "w") as f:
        f.write("# Transcript\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        for entry in transcript:
            f.write(f"**[{entry['time']}]** {entry['text']}\n\n")
    
    print(f"\nSaved to: {filename}")

class QuietHandler(SimpleHTTPRequestHandler):
    # HTTP handler that serves files and handles API calls
    
    def __init__(self, *args, app_state=None, **kwargs):
        self.app_state = app_state
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)
    
    def log_message(self, format, *args):
        pass  # suppressing HTTP logs
    
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/viewer.html"
        elif self.path == "/api/status":
            self.send_json({"recording": self.app_state.get("recording", False)})
            return
        super().do_GET()
    
    def do_POST(self):
        if self.path == "/api/start":
            self.app_state["recording"] = True
            self.app_state["start_event"].set()
            self.send_json({"status": "started"})
        elif self.path == "/api/stop":
            self.app_state["recording"] = False
            self.app_state["stop_event"].set()
            self.send_json({"status": "stopped"})
        else:
            self.send_error(404)
    
    def send_json(self, data):
        import json
        response = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

class ReuseAddrServer(HTTPServer):
    allow_reuse_address = True

def start_web_server(app_state):
    # starting web server
    handler = lambda *args, **kwargs: QuietHandler(*args, app_state=app_state, **kwargs)
    server = ReuseAddrServer(("", WEB_PORT), handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    return server

def main():
    global is_recording, transcript
    
    print("=" * 50)
    print("LIVE TRANSCRIBER")
    print("=" * 50)
    
    # shared state for web control
    app_state = {
        "recording": False,
        "start_event": threading.Event(),
        "stop_event": threading.Event(),
    }
    
    # initializing empty transcript JSON
    transcript = []
    save_transcript_json()
    
    # starting web server
    server = start_web_server(app_state)
    print(f"\nOpen in browser: http://localhost:{WEB_PORT}")
    
    # loading model
    print(f"\nLoading Whisper model '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Model loaded!")
    
    print("\n" + "=" * 50)
    print("Control recording from the web interface!")
    print("Or press Ctrl+C to quit.")
    print("=" * 50)
    
    # starting audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    
    try:
        stream.start()
        
        while True:
            # waiting for start signal from web
            print("\nWaiting for start from web interface...")
            app_state["start_event"].wait()
            app_state["start_event"].clear()
            
            # starting recording
            is_recording = True
            print("\nRECORDING...\n")
            
            # starting transcription worker
            worker = threading.Thread(target=transcription_worker, args=(model,))
            worker.daemon = True
            worker.start()
            
            # waiting for stop signal
            while not app_state["stop_event"].is_set():
                time.sleep(0.1)
            
            # stopping recording
            app_state["stop_event"].clear()
            is_recording = False
            app_state["recording"] = False
            print("\nStopped recording")
            save_transcript_md()
            
            # resetting transcript for next session
            transcript = []
            save_transcript_json()
    
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nShutting down...")
        is_recording = False
        stream.stop()
        stream.close()
        save_transcript_md()
        print("Goodbye!")

if __name__ == "__main__":
    main()