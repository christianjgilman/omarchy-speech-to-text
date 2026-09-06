# Dictation watcher — prints "1" while a capture stream matching the dictation
# app exists, "0" when it doesn't. Reacts to PipeWire routing events within
# ~30 ms; a slow heartbeat covers missed events. Hide is confirmed after only
# 60 ms of quiet so toggle-off feels instant without flicker on blips.
# Consumed by pill.qml (one long-lived process, change-only line output).
import re
import subprocess
import sys
import threading
import time

MATCH = re.compile(sys.argv[1] if len(sys.argv) > 1 else "wispr", re.I)
LOOP = 0.03          # main loop tick
HEARTBEAT = 0.25     # max gap between real checks even with no events
HIDE_AFTER = 0.06    # quiet time before confirming hide

got_event = threading.Event()


def pump_subscribe():
    proc = subprocess.Popen(
        ["pactl", "subscribe"], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True)
    for _ in proc.stdout:
        got_event.set()


threading.Thread(target=pump_subscribe, daemon=True).start()


def dictation_active():
    out = subprocess.run(
        ["pactl", "list", "source-outputs"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    return any(MATCH.search(b) for b in out.split("Source Output #")[1:])


state = None
last_check = 0.0
quiet_since = None
while True:
    now = time.monotonic()
    if got_event.is_set() or now - last_check >= HEARTBEAT:
        got_event.clear()
        last_check = now
        try:
            a = dictation_active()
        except Exception:
            time.sleep(LOOP)
            continue

        if a != state:
            if state is None:  # first reading: emit immediately
                state = a
                quiet_since = None
                print("1" if state else "0", flush=True)
            elif a:  # show immediately
                state = True
                quiet_since = None
                print("1", flush=True)
            else:  # hide only after brief confirmed quiet
                if quiet_since is None:
                    quiet_since = now
                elif now - quiet_since >= HIDE_AFTER:
                    state = False
                    quiet_since = None
                    print("0", flush=True)
        else:
            quiet_since = None
    time.sleep(LOOP)
