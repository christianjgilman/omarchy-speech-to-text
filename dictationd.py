#!/usr/bin/env python3
"""dictationd — three-mode dictation daemon (SenseVoice offline engine).

Modes (controlled via unix socket, see dictationctl):
  send  : record; text accrues silently. Ctrl+Tab stop = paste + Enter (auto-send).
          Ctrl+Caps mid-send = paste the section so far, keep recording.
  live  : always-on transcription — each spoken phrase is typed at the cursor
          right after its pause (Silero VAD segmentation, ~0.5s of silence).
          Silence produces nothing (no phantom words).

Engine: SenseVoice small int8, offline decode per VAD-delimited segment
(RTF ~0.07 on this machine) — outputs punctuation + capitalization.
Capture stream is named "dictationd" via PIPEWIRE_PROPS so the
dictation-pill plugin can gate on it.
"""

import ctypes
import collections
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sherpa_onnx

# --- tuning -----------------------------------------------------------------
MODEL_DIR = os.path.expanduser("~/.local/share/dictationd/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
VAD_MODEL = os.path.expanduser("~/.local/share/dictationd/models/silero_vad.onnx")
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.42         # silero speech probability gate (lower = hears soft fillers)
MIN_SPEECH_MS = 120           # discard blips shorter than this (kept low: soft fillers)
PRE_BUFFER_S = 0.40           # audio kept before speech starts (word onsets)
SEGMENT_END_SILENCE_S = 0.7   # silence that closes a segment (triggers decode)
MAX_SEGMENT_S = 10.0          # force a segment break on very long continuous speech
HTTP_HOST = "127.0.0.1"       # OpenAI-compatible STT endpoint (local only)
HTTP_PORT = 8765
LIVE = "live"
SEND = "send"

decode_lock = threading.Lock()  # SenseVoice decode is not thread-safe

SOCK_PATH = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000"), "dictationd.sock")


def log(*a):
    print(*a, flush=True)


def prctl_name(name):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(15, name.encode(), 0, 0, 0)  # PR_SET_NAME


def wtype(*args):
    try:
        subprocess.run(["wtype", *args], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        log("wtype failed:", e)
        return False


def type_text(text):
    if text:
        wtype(text)


def start_delta(text):
    if text and text[0].isalpha():
        return text[0].upper() + text[1:]
    return text


TERMINAL_CLASSES = {"alacritty", "foot", "kitty", "ghostty", "wezterm",
                    "org.wezfurlong.wezterm", "konsole", "xterm-256color"}
RESTORE_CLIPBOARD_S = 0.6  # how long the target has to read the paste


def focused_window_class():
    """Best-effort focused Hyprland window class (for terminal paste variant)."""
    try:
        out = subprocess.run(["hyprctl", "-j", "activewindow"], capture_output=True,
                             text=True, timeout=2).stdout
        import json as _json
        return (_json.loads(out).get("class") or "").lower()
    except Exception:
        return ""


def paste_text(text, enter=False):
    """Deliver text by clipboard + Ctrl+V (instant, no keystroke simulation).
    Terminals get Ctrl+Shift+V. The previous clipboard is restored afterwards."""
    if not text and not enter:
        return
    if text:
        cls = focused_window_class()
        prev = None
        try:
            prev = subprocess.run(["wl-paste", "--no-newline"], capture_output=True,
                                  text=True, timeout=2).stdout
        except Exception:
            pass
        subprocess.run(["wl-copy", text], check=True)
        if cls in TERMINAL_CLASSES:
            wtype("-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl")
        else:
            wtype("-M", "ctrl", "-k", "v", "-m", "ctrl")
        log(f"paste: {len(text)} chars -> focused='{cls}' enter={enter}")
        if prev is not None:
            def restore():
                time.sleep(RESTORE_CLIPBOARD_S)
                try:
                    if prev == "":
                        subprocess.run(["wl-copy", "--clear"], check=False)
                    else:
                        subprocess.run(["wl-copy", prev], check=False)
                except Exception:
                    pass
            threading.Thread(target=restore, daemon=True).start()
    if enter:
        wtype("-k", "Return")


# --- engine -----------------------------------------------------------------
log("loading models...")
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=f"{MODEL_DIR}/model.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    num_threads=2,
    sample_rate=SAMPLE_RATE,
    use_itn=True,
    language="en")

silero_cfg = sherpa_onnx.SileroVadModelConfig(
    model=VAD_MODEL, threshold=VAD_THRESHOLD,
    min_speech_duration=MIN_SPEECH_MS / 1000.0, min_silence_duration=0.4)
vad_cfg = sherpa_onnx.VadModelConfig()
vad_cfg.silero_vad = silero_cfg
vad_cfg.provider = "cpu"
vad = sherpa_onnx.VadModel.create(vad_cfg)
vad_window = vad.window_size()  # samples silero expects per is_speech call
END_SILENCE_WINDOWS = int(SEGMENT_END_SILENCE_S * SAMPLE_RATE / vad_window)
MAX_SEGMENT_WINDOWS = int(MAX_SEGMENT_S * SAMPLE_RATE / vad_window)
log("models ready")


def decode_windows(windows):
    """Offline-decode concatenated float32 windows; returns styled text."""
    if not windows:
        return ""
    audio = np.concatenate(windows)
    if len(audio) < SAMPLE_RATE * 0.35:
        return ""
    with decode_lock:
        s = rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, audio.tolist())
        rec.decode_stream(s)
        return s.result.text.strip()


class Session:
    def __init__(self, mode):
        self.mode = mode              # SEND | LIVE
        self.windows = []             # speech windows of the open segment
        self.pending_text = ""        # decoded text not yet typed (send mode)
        self.pending_has_text = False
        self.speech_windows = 0       # decaying speech counter for the gate
        self.speech_seen = False
        self.prebuf = collections.deque(maxlen=int(PRE_BUFFER_S * SAMPLE_RATE / vad_window))
        self.silence_run = 0
        self.queue = collections.deque()
        self.mic = None


state_lock = threading.Lock()
session = None
stop_flag = threading.Event()


class MicStream:
    """pw-record native capture; pushes silero-sized windows to the worker.

    Uses a PipeWire-native client so the stream is named "dictationd"
    (via PIPEWIRE_PROPS) and shows up in the pill's match pattern.
    """

    def __init__(self, on_window):
        self.on_window = on_window
        self.proc = None
        self.reader = None

    def _env(self):
        env = dict(os.environ)
        env["PIPEWIRE_PROPS"] = '{ "application.name" = "dictationd" }'
        return env

    def start(self):
        # safety: kill any pw-record this machine left behind earlier
        subprocess.run(["pkill", "-x", "pw-record"], stderr=subprocess.DEVNULL)
        self.proc = subprocess.Popen(
            ["pw-record", "--raw", "--rate", str(SAMPLE_RATE),
             "--channels", "1", "--format", "f32", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=self._env())

        def pump():
            while True:
                raw = self.proc.stdout.read(vad_window * 4)
                if not raw or len(raw) < vad_window * 4:
                    break
                self.on_window(np.frombuffer(raw, dtype=np.float32).copy())

        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()

    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def emit_segment(s, force=False):
    """Close the open segment: decode its audio. LIVE types immediately;
    SEND banks the text into pending_text (typed on flush/stop)."""
    text = decode_windows(s.windows)
    s.windows = []
    s.silence_run = 0
    if not text:
        return
    if s.mode == LIVE:
        type_text(text + " ")
    elif s.pending_text:
        s.pending_text += " " + text
    else:
        s.pending_text = text


def worker():
    global session
    while not stop_flag.is_set():
        s = session
        win = None
        if s is not None:
            if s.queue:
                win = s.queue.popleft()
            else:
                time.sleep(0.01)
                continue
        else:
            time.sleep(0.05)
            continue
        try:
            is_speech = vad.is_speech(win.tolist())
        except Exception as e:
            log("vad error:", e)
            continue

        if not s.speech_seen:
            s.prebuf.append(win)
            # decaying counter: word gaps don't reset it, random blips don't fire it
            if is_speech:
                s.speech_windows += 1
            else:
                s.speech_windows = max(0, s.speech_windows - 1)
            if s.speech_windows >= 2:
                s.speech_seen = True
                log("speech detected -> decoding")
                for w in list(s.prebuf):
                    s.windows.append(w)
                s.prebuf.clear()
            continue

        # inside a speech segment
        s.windows.append(win)
        if not is_speech:
            s.silence_run += 1
            if s.silence_run >= END_SILENCE_WINDOWS:
                emit_segment(s)
                s.speech_seen = False
                s.prebuf.clear()
                s.prebuf.append(win)  # keep trailing silence as new pre-buffer
        else:
            s.silence_run = 0
            if sum(len(w) for w in s.windows) >= MAX_SEGMENT_WINDOWS * vad_window:
                emit_segment(s)
                s.speech_seen = True  # still talking; keep segment open


def start_session(mode):
    global session
    with state_lock:
        if session is not None:
            return f"busy:{session.mode}"
        session = Session(mode)
        mic = MicStream(lambda w: session.queue.append(w) if session else None)
        session.mic = mic
        mic.start()
        log(f"session start: {mode}")
        return f"started:{mode}"


def stop_session(auto_enter=False):
    """Finalize gracefully: drain the capture tail, decode everything pending,
    type it, close mic. Enter only when auto_enter (Ctrl+Tab send stop)."""
    global session
    with state_lock:
        s = session
        if s is None:
            return "idle"
    # drain: keep the session alive so the worker ingests the tail audio
    deadline = time.monotonic() + 0.45
    while time.monotonic() < deadline:
        if not s.queue and s.silence_run >= END_SILENCE_WINDOWS:
            break
        time.sleep(0.02)
    with state_lock:
        session = None
    s.mic.stop()
    log(f"session stop: {s.mode}")
    # decode whatever segment is still open, banked into pending
    text = decode_windows(s.windows)
    if text:
        s.pending_text = (s.pending_text + " " + text) if s.pending_text else text
    if s.pending_text:
        paste_text(s.pending_text)
    if auto_enter:
        wtype("-k", "Return")
    return f"stopped:{s.mode}"


def flush_segment():
    """Send mode: decode + paste the section so far, keep recording."""
    with state_lock:
        s = session
        if s is None or s.mode != SEND:
            log(f"flush: not in send mode (session={s.mode if s else None})")
            return "not-send"
        open_windows = len(s.windows)
        banked = len(s.pending_text)
    log(f"flush: open_windows={open_windows} banked_chars={banked}")
    emit_segment(s)  # close the open segment into pending
    if s.pending_text:
        paste_text(s.pending_text + " ")
        s.pending_text = ""
        return "flushed"
    log("flush: nothing buffered")
    return "empty"


def handle(cmd):
    if cmd == "send":
        with state_lock:
            cur = session.mode if session else None
        if cur == SEND:
            return stop_session(auto_enter=True)
        if cur == LIVE:
            return stop_session(auto_enter=False)
        return start_session(SEND)
    if cmd == "caps":
        with state_lock:
            cur = session.mode if session else None
        if cur == SEND:
            return flush_segment()
        if cur == LIVE:
            return stop_session(auto_enter=False)
        return start_session(LIVE)
    if cmd == "status":
        with state_lock:
            cur = session.mode if session else None
        return cur or "idle"
    if cmd == "stop":
        return stop_session(auto_enter=False)
    return f"unknown:{cmd}"


# --- OpenAI-compatible STT endpoint -----------------------------------------
def extract_multipart_file(body, content_type):
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or "")
    if not m:
        return None
    boundary = (m.group(1) or m.group(2)).strip().encode()
    for part in body.split(b"--" + boundary):
        if part[:2] == b"\n":
            part = part[2:]
        if part in (b"", b"--") or part.startswith(b"--"):
            continue
        hdr_end = part.find(b"\r\n\r\n")
        if hdr_end == -1:
            continue
        headers = part[:hdr_end]
        if b'name="file"' not in headers and b"filename=" not in headers:
            continue
        content = part[hdr_end + 4:]
        if content.endswith(b"\r\n"):
            content = content[:-2]
        return content
    return None


def to_pcm16k_f32(data):
    """Convert any audio container/format to 16k mono float32 via ffmpeg."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
        input=data, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


class SttHandler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/health":
            self._json(200, {"status": "ok", "engine": "sensevoice-small-int8",
                             "modes": ["send", "live"]})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "sensevoice-small-int8", "object": "model", "owned_by": "dictationd"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/audio/transcriptions":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = extract_multipart_file(body, self.headers.get("Content-Type", ""))
            if data is None:
                self._json(400, {"error": "no file part in multipart body"})
                return
            pcm = to_pcm16k_f32(data)
            if pcm is None or len(pcm) < SAMPLE_RATE // 10:
                self._json(400, {"error": "could not decode audio (ffmpeg)"})
                return
            with decode_lock:
                stream = rec.create_stream()
                stream.accept_waveform(SAMPLE_RATE, pcm.tolist())
                rec.decode_stream(stream)
                text = stream.result.text.strip()
            self._json(200, {"text": text})
        except Exception as e:
            log("http error:", e)
            self._json(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def serve_http():
    try:
        srv = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), SttHandler)
        log(f"OpenAI-compatible STT endpoint: http://{HTTP_HOST}:{HTTP_PORT}/v1/audio/transcriptions")
        srv.serve_forever()
    except Exception as e:
        log("http server failed:", e)


def serve():
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o600)
    srv.listen(4)
    log("listening on", SOCK_PATH)
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                data = conn.recv(256).decode().strip()
                if data:
                    conn.sendall(handle(data).encode())
            except Exception as e:
                log("conn error:", e)


def selftest(wav_path):
    import wave
    w = wave.open(wav_path)
    sr = w.getframerate()
    samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(samples) / sr
    s = rec.create_stream()
    s.accept_waveform(sr, samples)
    t0 = time.perf_counter()
    rec.decode_stream(s)
    dt = time.perf_counter() - t0
    print(f"audio {dur:.1f}s | decode {dt:.2f}s | RTF {dt/dur:.3f}")
    print("text:", s.result.text.strip())


def _cleanup(*_):
    try:
        os.unlink(SOCK_PATH)
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    prctl_name("dictationd")
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    if len(sys.argv) > 2 and sys.argv[1] == "selftest":
        selftest(sys.argv[2])
        sys.exit(0)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=serve_http, daemon=True).start()
    log("dictationd ready")
    serve()
