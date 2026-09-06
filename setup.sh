#!/usr/bin/env bash
# setup.sh — idempotent installer/runner for the dictationd engine.
#   ./setup.sh ensure    install venv + models + ctl if missing (fast no-op when present)
#   ./setup.sh daemon    ensure deps, then exec the daemon (used by the plugin service)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/.local/share/dictationd/venv"
MODELS="$HOME/.local/share/dictationd/models"
PY="$VENV/bin/python"

ensure_deps() {
	if [ ! -x "$PY" ]; then
		echo "dictationd: creating venv..."
		python3 -m venv "$VENV"
		"$VENV/bin/pip" install -q sherpa-onnx numpy
	fi
	mkdir -p "$MODELS" "$HOME/.local/bin"
	if [ ! -f "$MODELS/silero_vad.onnx" ]; then
		echo "dictationd: downloading silero VAD..."
		curl -sL -o "$MODELS/silero_vad.onnx" \
			https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
	fi
	SV="$MODELS/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
	if [ ! -f "$SV/model.int8.onnx" ]; then
		echo "dictationd: downloading SenseVoice small int8 (~240MB)..."
		curl -sL -o "$MODELS/sv.tar.bz2" \
			https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
		tar xf "$MODELS/sv.tar.bz2" -C "$MODELS" && rm -f "$MODELS/sv.tar.bz2"
	fi
	if [ ! -e "$HOME/.local/bin/dictationctl" ] || ! cmp -s "$DIR/dictationctl" "$HOME/.local/bin/dictationctl"; then
		cp "$DIR/dictationctl" "$HOME/.local/bin/dictationctl"
		chmod +x "$HOME/.local/bin/dictationctl"
	fi
}

case "${1:-ensure}" in
ensure)
	ensure_deps
	echo "dictationd: deps ready"
	;;
daemon)
	ensure_deps
	export PIPEWIRE_PROPS='{ "application.name" = "dictationd" }'
	exec "$PY" "$DIR/dictationd.py"
	;;
*)
	echo "usage: setup.sh [ensure|daemon]" >&2
	exit 2
	;;
esac
