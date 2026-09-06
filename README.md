# Omarchy Speech to Text

Local, real-time speech-to-text for [Omarchy](https://omarchy.org/) — a
streaming dictation daemon + a minimal "dictation pill" overlay that shows
you're being heard. Fully offline, fully yours.

## The three modes

| Keys | Mode | Behavior |
|------|------|----------|
| `Ctrl+Tab` → talk → `Ctrl+Tab` | **Send** | Stop transcribes everything and pastes it, then presses **Enter** (auto-send). |
| `Ctrl+Tab` → talk → `Ctrl+grave` (each section) → `Ctrl+Tab` | **Rant** | Every `Ctrl+grave` instantly pastes the section you just spoke and keeps recording. Final `Ctrl+Tab` lands the tail + Enter. Built for brain dumps. |
| `Ctrl+grave` → talk → `Ctrl+grave` | **Live** | Transcription types at your cursor phrase-by-phrase as you talk, for as long as you like. Silence produces nothing (no phantom words). `Ctrl+grave` stops gracefully. |

Delivery is by **clipboard paste** (Ctrl+V), not keystroke simulation — one
atomic insertion, terminals automatically get Ctrl+Shift+V, and your previous
clipboard content is restored after each paste.

## The engine

- **SenseVoice small int8** (offline, English-first) decoding Silero-VAD
  segments — RTF ~0.07 on an i7-6700HQ. Punctuation and capitalization included.
- **Anti-hallucination**: the recognizer only ever sees audio the VAD marked
  as speech (with a pre-buffer so word onsets survive). Silence cannot decode.
- **No repeats**: typed output is delta-tracked per stream.
- **OpenAI-compatible STT endpoint**: `POST http://127.0.0.1:8765/v1/audio/transcriptions`
  (multipart `file`, any format — ffmpeg converts). Any tool that speaks the
  OpenAI audio API can use your local engine:

```bash
curl -F file=@audio.webm http://127.0.0.1:8765/v1/audio/transcriptions
# {"text": "..."}
```

The pill overlay matches the capture stream, so it lights up for this daemon
*and* anything else you add to its match pattern (see `DictationPill.qml`).

## Install

```bash
omarchy plugin add https://github.com/christianjgilman/omarchy-speech-to-text --enable
```

Then add your keybindings in `~/.config/hypr/bindings.lua`:

```lua
hl.unbind("CTRL + TAB")  # if bound (Omarchy default: window-capture picker)
o.bind("CTRL + TAB", "Toggle dictation (send)", "dictationctl send")
o.bind("CTRL + grave", "Live dictation / flush section", "dictationctl caps")
```

First run bootstraps itself: venv, sherpa-onnx, Silero VAD, and the SenseVoice
model (~240MB) — after that it's instant at every login.

## Requirements

- Omarchy 4 / Hyprland + Quickshell, `ffmpeg`, `python3`, `wtype`, `wl-clipboard`

## Tuning

All in `dictationd.py`: `VAD_THRESHOLD` (ghost-word vs missed-fillers),
`MIN_SPEECH_MS`, `PRE_BUFFER_S`, `SEGMENT_END_SILENCE_S` (phrase latency),
`MAX_SEGMENT_S`, and the endpoint port (`HTTP_PORT`, default 8765, localhost only).

## Uninstall

```bash
omarchy plugin remove champion.speech-to-text
rm -rf ~/.local/share/dictationd   # venv + models
```

## License

MIT

## Roadmap

- Settings UI as a proper Omarchy panel plugin (bar icon + popup, integrated with the Omarchy menu system) — the daemon's HTTP API (`/api/words`, `/settings`) is already the backend for it.
- Custom-word tuning over time: the replacements live in `~/.config/dictationd/custom-words.json` — trivially editable by hand, script, or an AI agent session.
- Optional: model-level hotword biasing via a transducer engine (Parakeet/hotwords-file), if a compatible runtime lands.
