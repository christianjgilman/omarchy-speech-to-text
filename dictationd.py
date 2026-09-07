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
import difflib
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
VAD_THRESHOLD = 0.50         # silero speech probability gate (lower = hears soft fillers, higher = ignores music; ghost single-words are caught by the one-word filter)
MIN_SPEECH_MS = 120           # discard blips shorter than this (kept low: soft fillers)
PRE_BUFFER_S = 0.40           # audio kept before speech starts (word onsets)
SEGMENT_END_SILENCE_S = 1.0   # silence that closes a segment (triggers decode; high enough to ride over mid-phrase pauses)
SPLIT_SOFT_S = 10.0           # start hunting for a word gap to split long speech
SPLIT_HARD_S = 14.0           # cut by now even mid-word, at the quietest recent window
SPLIT_DIP_WINDOWS = 2         # consecutive quiet windows that count as a word gap (~64ms)
SPLIT_LOOKBACK_S = 1.5        # window searched for the quietest cut point on hard split
FUZZY_WORD_RATIO = 0.72       # vocab entries: near-miss transcript words corrected to the target
ONE_WORD_WHITELIST = {"launch", "go", "approved", "bro", "it's", "i'm", "don't", "can't", "won't", "you're", "we're", "they're", "isn't", "doesn't", "didn't", "c", "ci"}  # single-word segments kept only if in this set (lowercase)
HTTP_HOST = "127.0.0.1"       # OpenAI-compatible STT endpoint (local only)
HTTP_PORT = 8765
LIVE = "live"
SEND = "send"

decode_lock = threading.Lock()  # SenseVoice decode is not thread-safe

SOCK_PATH = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000"), "dictationd.sock")
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_WORDS_PATH = os.path.expanduser("~/.config/dictationd/custom-words.json")


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


paste_lock = threading.Lock()  # copy+paste must be atomic vs clipboard restore


def paste_now(text):
    """wl-copy + paste keystroke as one atomic unit (paste_lock held)."""
    subprocess.run(["wl-copy", text], check=True)
    cls = focused_window_class()
    if cls in TERMINAL_CLASSES:
        wtype("-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl")
    else:
        wtype("-M", "ctrl", "-k", "v", "-m", "ctrl")
    return cls


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
        with paste_lock:
            cls = paste_now(text)
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
SPLIT_SOFT_WINDOWS = int(SPLIT_SOFT_S * SAMPLE_RATE / vad_window)
SPLIT_HARD_WINDOWS = int(SPLIT_HARD_S * SAMPLE_RATE / vad_window)
SPLIT_LOOKBACK_WINDOWS = int(SPLIT_LOOKBACK_S * SAMPLE_RATE / vad_window)
SENTENCE_ENDS = ".!?\u2026" 
log("models ready")


def load_custom_words():
    """Return the replacements list ([{from,to}, ...]); missing file = empty."""
    try:
        with open(CUSTOM_WORDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        words = data.get("replacements", [])
        return [w for w in words if w.get("to")]
    except Exception:
        return []


def save_custom_words(words):
    os.makedirs(os.path.dirname(CUSTOM_WORDS_PATH), exist_ok=True)
    with open(CUSTOM_WORDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"replacements": words}, f, ensure_ascii=False, indent=1)


def apply_one_word_filter(text):
    """Drop one-word segments unless whitelisted (music/ghost artifacts
    surface as lone words; real commands are whitelisted)."""
    if text and len(text.split()) == 1 and text.lower().strip(".,!?;:") not in ONE_WORD_WHITELIST:
        log("dropped one-word segment:", text)
        return ""
    return text


SKILL_DIRS = ("~/.agents/skills", "~/.zcode/skills")
SLASH_WORD = "slash"
SLASH_RATIO = 0.8  # fuzzy floor for matching a spoken name to a real skill


def load_slash_skills():
    """Live skill names (refreshed every decode; globbing two dirs is cheap)."""
    names = set()
    for d in SKILL_DIRS:
        base = os.path.expanduser(d)
        try:
            for entry in os.listdir(base):
                if entry[0].isalnum() and os.path.isdir(os.path.join(base, entry)):
                    names.add(entry.lower())
        except OSError:
            continue
    return names


def match_skill(spoken, skills):
    """Exact match always wins; otherwise fuzzy >= SLASH_RATIO for real tries."""
    if spoken in skills:
        return spoken
    if len(spoken) < 4:
        return None
    best, best_r = None, 0.0
    for s in skills:
        r = difflib.SequenceMatcher(None, spoken, s).ratio()
        if r > best_r:
            best, best_r = s, r
    return best if best_r >= SLASH_RATIO else None


def apply_slash_commands(text):
    """Spoken commands: "slash speech" -> "/speech". The name after "slash"
    must match a real zcode skill (1-3 tokens joined, kebab-aware), otherwise
    the text is left alone. "slash" never converts to "/" by itself."""
    tokens = text.split(" ")
    if SLASH_WORD not in [t.lower().strip('.,!?;:"') for t in tokens]:
        return text
    skills = load_slash_skills()
    if not skills:
        return text
    out, i = [], 0
    while i < len(tokens):
        core = tokens[i].strip('.,!?;:"').lower()
        if core == SLASH_WORD and i + 1 < len(tokens):
            hit, span = None, 0
            for n in (1, 2, 3):
                if i + 1 + n > len(tokens):
                    break
                spoken = "".join(tokens[i + 1 + j].strip('.,!?;:"').lower()
                                 for j in range(n))
                m = match_skill(spoken, skills)
                if m:
                    hit, span = m, n
                    break
            if hit:
                log("slash-command:", "/{}".format(hit))
                out.append("/" + hit)
                i += 1 + span  # "slash" + the name tokens
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def apply_custom_words(text):
    """Two layers:
    1. exact replacements (heard -> write), case-insensitive, longest first;
    2. vocabulary entries (write-only): near-miss transcript words/phrases
       fuzzy-matched to the target, so you can add names without knowing what
       the recognizer mangles them into."""
    text = apply_one_word_filter(text)
    text = apply_slash_commands(text)
    words = load_custom_words()
    if not text or not words:
        return text
    for w in sorted([w for w in words if w.get("from")],
                    key=lambda w: len(w["from"]), reverse=True):
        # whole-word only: a short "from" ("c") must never fire inside a
        # real word ("appreciate")
        pattern = r"(?<!\w)" + re.escape(w["from"]) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            log("custom-word:", w["from"], "->", w["to"])
        text = re.sub(pattern, w["to"], text, flags=re.IGNORECASE)

    vocab = [w["to"].strip() for w in words if not w.get("from") and len(w["to"].strip()) >= 4]
    if not vocab:
        return text
    tokens = text.split(" ")
    for target in vocab:
        t_lower = target.lower()
        t_compact = t_lower.replace(" ", "")
        t_tokens = t_lower.split(" ")
        n = len(t_tokens)
        i = 0
        while i < len(tokens):
            core = tokens[i].strip('.,!?;:"').lower()
            if n == 1:
                # two-token window first ("moto gp"), then single token
                # ("valen"); single-token first would eat half of the
                # bigram ("MotoGP gp") and leave a stray token behind.
                hit = False
                span = 1
                if i + 1 < len(tokens):
                    nxt = tokens[i + 1].strip('.,!?;:"').lower()
                    bigram = core + nxt
                    if (bigram == t_compact or (t_lower not in bigram and len(bigram) >= 6)) and difflib.SequenceMatcher(None, bigram, t_compact).ratio() >= FUZZY_WORD_RATIO:
                        hit = True
                        span = 2
                if not hit:
                    hit = len(core) >= 4 and difflib.SequenceMatcher(None, core, t_lower).ratio() >= FUZZY_WORD_RATIO
                    span = 1
            else:
                window = tokens[i:i + n]
                if len(window) < n:
                    i += 1
                    continue
                cand = "".join(x.strip('.,!?;:"').lower() for x in window)
                hit = len(cand) >= 6 and difflib.SequenceMatcher(None, cand, t_compact).ratio() >= FUZZY_WORD_RATIO
                span = n
            if hit:
                log("vocab:", " ".join(tokens[i:i + span]), "->", target)
                tokens[i:i + span] = [target]
                i += span
                continue
            i += 1
    return " ".join(tokens)
    return " ".join(tokens)


def log_transcript(text, source):
    """Append every decoded phrase to one JSONL log for later analysis."""
    import datetime, json
    entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
             "source": source, "text": text}
    with open(os.path.expanduser("~/.local/share/dictationd/transcripts.jsonl"), "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def decode_windows(windows, source="dictation"):
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
        text = apply_custom_words(s.result.text.strip())
    if text:
        try:
            log_transcript(text, source)
        except OSError as e:
            log("transcript log write failed:", e)
    return text


class Session:
    def __init__(self, mode):
        self.mode = mode              # SEND | LIVE
        self.windows = []             # speech windows of the open segment
        self.win_energy = []          # per-window RMS, parallel to windows
        self.tail = ""                # last char of the last pasted segment (for casing)
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


def deliver(s, text, source):
    """Case a decoded segment against what was said before it, then deliver:
    LIVE pastes immediately; SEND banks into pending_text."""
    if not text:
        return
    if s.tail and s.tail[-1] not in SENTENCE_ENDS:
        text = text[:1].lower() + text[1:]
    s.tail = text.rstrip()[-1:] or s.tail
    if s.mode == LIVE:
        paste_text(text + " ")
    elif s.pending_text:
        s.pending_text += " " + text
    else:
        s.pending_text = text


def emit_segment(s, force=False):
    """Close the open segment: decode its audio and clear it."""
    text = decode_windows(s.windows, source=s.mode)
    s.windows = []
    s.win_energy = []
    s.silence_run = 0
    deliver(s, text, s.mode)


def split_segment(s, cut):
    """Best-cut a long segment at a word boundary: decode the head, keep the
    tail (and its energies) as the start of the next segment. Speech continues
    without the gate re-arming, so the flow never stutters."""
    head, tail = s.windows[:cut], s.windows[cut:]
    ehead, etail = s.win_energy[:cut], s.win_energy[cut:]
    s.windows, s.win_energy = tail, etail
    s.silence_run = 0
    text = decode_windows(head, source=f"{s.mode}-split")
    deliver(s, text, s.mode)


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

        # one bad window or a bug in a new feature must NEVER kill the
        # pipeline: a dead worker black-holes all dictation until restart
        # (happened 2026-09-07: slash-command IndexError silenced live mode
        # mid-session, words only surfaced on session stop)
        try:
            worker_step(s, win, is_speech)
        except Exception as e:
            import traceback
            log("worker step error:", e)
            traceback.print_exc()

def worker_step(s, win, is_speech):
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
            return

        # inside a speech segment
        s.windows.append(win)
        s.win_energy.append(float(np.sqrt(np.mean(win * win))))
        total = sum(len(w) for w in s.windows)
        if not is_speech:
            if s.silence_run == 0:
                s.dip_start = len(s.windows) - 1
            s.silence_run += 1
            if s.silence_run >= END_SILENCE_WINDOWS:
                emit_segment(s)
                s.speech_seen = False
                s.prebuf.clear()
                s.prebuf.append(win)  # keep trailing silence as new pre-buffer
            elif s.silence_run >= SPLIT_DIP_WINDOWS and total >= SPLIT_SOFT_WINDOWS * vad_window:
                # natural word gap in long speech: cut at the dip start
                split_segment(s, s.dip_start)
        else:
            s.silence_run = 0
            if total >= SPLIT_HARD_WINDOWS * vad_window:
                # truly continuous speech, no gap found: cut at the quietest
                # window of the recent tail (least-bad word boundary)
                look = min(SPLIT_LOOKBACK_WINDOWS, len(s.win_energy) - 1)
                cut = len(s.win_energy) - look + int(np.argmin(s.win_energy[-look:]))
                split_segment(s, cut)
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
    """Reply immediately; drain the tail + decode + paste in the background so
    the compositor (which dispatched the keypress) never blocks on a decode."""
    global session
    with state_lock:
        s = session
        if s is None:
            return "idle"
        session = None
    threading.Thread(target=_finalize, args=(s, auto_enter), daemon=True).start()
    return f"stopping:{s.mode}"


def _finalize(s, auto_enter):
    """Background finalize: capture the tail, decode the open segment, paste."""
    deadline = time.monotonic() + 0.35
    while time.monotonic() < deadline:
        if s.queue:
            time.sleep(0.02)
        else:
            time.sleep(0.05)
            if not s.queue:
                break
    while s.queue:
        s.windows.append(s.queue.popleft())
    s.mic.stop()
    log(f"session stop: {s.mode}")
    text = decode_windows(s.windows, source=f"{s.mode}-tail")
    if text:
        s.pending_text = (s.pending_text + " " + text) if s.pending_text else text
    if s.pending_text:
        paste_text(s.pending_text)
    if auto_enter:
        wtype("-k", "Return")


def flushenter_live():
    """LIVE mode: decode + paste everything pending + press Enter, keep
    recording so the next phrase starts with zero latency."""
    with state_lock:
        s = session
        if s is None or s.mode != LIVE:
            return "not-live"
        windows = s.windows
        s.windows = []
        s.silence_run = 0

    def _flush():
        text = decode_windows(windows, source="live-flushenter")
        if text:
            paste_text(text + " ")
        wtype("-k", "Return")

    threading.Thread(target=_flush, daemon=True).start()
    return "flushing"


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
    windows = s.windows
    s.windows = []
    s.silence_run = 0

    def _flush():
        text = decode_windows(windows, source=f"{s.mode}-flush")
        if text:
            s.pending_text = (s.pending_text + " " + text) if s.pending_text else text
            paste_text(s.pending_text + " ")
            s.pending_text = ""

    threading.Thread(target=_flush, daemon=True).start()
    return "flushing"


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
    if cmd == "flushenter":
        return flushenter_live()
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
        elif path == "/settings":
            try:
                with open(os.path.join(PLUGIN_DIR, "settings.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._json(500, {"error": str(e)})
        elif path == "/api/words":
            self._json(200, {"words": load_custom_words()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/api/words":
            length = int(self.headers.get("Content-Length", 0))
            try:
                item = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._json(400, {"error": "bad json"})
                return
            frm, to = str(item.get("from", "")).strip(), str(item.get("to", "")).strip()
            if not to or (frm and frm.lower() == to.lower()):
                self._json(400, {"error": "to is required (from optional)"})
                return
            words = load_custom_words()
            if frm:
                words = [w for w in words if (w.get("from") or "").lower() != frm.lower()]
            else:
                words = [w for w in words if w.get("to") != to]
            words.append({"from": frm, "to": to})
            save_custom_words(words)
            self._json(200, {"ok": True, "words": words})
            return
        if path == "/api/words/delete":
            length = int(self.headers.get("Content-Length", 0))
            try:
                item = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._json(400, {"error": "bad json"})
                return
            frm = str(item.get("from", "")).strip().lower()
            words = [w for w in load_custom_words() if (w.get("from") or "").lower() != frm]
            save_custom_words(words)
            self._json(200, {"ok": True, "words": words})
            return
        if path != "/v1/audio/transcriptions":
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
                text = apply_custom_words(stream.result.text.strip())
            if text:
                try:
                    log_transcript(text, "api")
                except OSError as e:
                    log("transcript log write failed:", e)
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
