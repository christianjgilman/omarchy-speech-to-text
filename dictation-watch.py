# Dictation watcher — prints "1" while a capture stream matching the dictation
# app exists, "0" when it doesn't. Reacts to PipeWire routing events within
# ~30 ms; a slow heartbeat covers missed events. Hide is confirmed after only
# 60 ms of quiet so toggle-off feels instant without flicker on blips.
# Consumed by pill.qml (one long-lived process, change-only line output).
#
# SINGLETON LAW (2026-09-09): the shell spawns a fresh copy per start without
# killing the previous one, and copies piled up as orphans — six copies
# polling pactl 4x/s cost ~1.4 cores of audio-daemon CPU. Every startup now
# reaps older copies (and their subscribe children) first, and a TERM/INT
# handler kills this copy's own subscribe so nothing orphans on exit.
import os
import re
import signal
import subprocess
import sys
import threading
import time

MATCH = re.compile(sys.argv[1] if len(sys.argv) > 1 else "wispr", re.I)
LOOP = 0.03          # main loop tick
HEARTBEAT = 0.25     # max gap between real checks even with no events
HIDE_AFTER = 0.06    # quiet time before confirming hide

SCRIPT = os.path.basename(sys.argv[0])
subscribe_proc = None

got_event = threading.Event()


def _stat_fields(pid):
    # fields after "(comm)": state, ppid, ... — split on the closing paren
    # so a comm containing spaces can't shift the indexes
    with open(f"/proc/{pid}/stat") as f:
        return f.read().rsplit(")", 1)[1].split()


def _start_time(pid):
    # field 22 from stat start (index 19 after comm): clock ticks at spawn
    return int(_stat_fields(pid)[19])


def _children(pid):
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                if int(_stat_fields(int(p))[1]) == pid:
                    out.append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    return out


def reap_older_copies():
    my_start = _start_time(os.getpid())
    skip = {os.getpid(), os.getppid()}
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in skip:
            continue
        pid = int(p)
        try:
            if open(f"/proc/{pid}/comm").read().strip() != "python3":
                continue
            if not any(a.endswith(SCRIPT)
                       for a in open(f"/proc/{pid}/cmdline").read().split("\0")):
                continue
            # strictly older only: same-tick copies must never kill each
            # other, a rare duplicate costs less than zero watchers
            if _start_time(pid) >= my_start:
                continue
        except (OSError, ValueError, IndexError):
            continue
        for child in _children(pid):
            try:
                os.kill(child, signal.SIGTERM)
            except OSError:
                pass
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def reap_orphan_subscribes():
    # A SIGKILLed watcher (shell reloads do this) can't run the exit guard,
    # orphaning its subscribe to init/systemd. Only watchers ever spawn
    # "pactl subscribe" here, so a subscribe whose parent is not a live
    # watcher copy is a stray: reap it at every startup.
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        pid = int(p)
        try:
            argv = open(f"/proc/{pid}/cmdline").read().split("\0")
            if not (os.path.basename(argv[0] or "") == "pactl"
                    and len(argv) > 1 and argv[1] == "subscribe"):
                continue
            ppid = int(_stat_fields(pid)[1])
            pcomm = open(f"/proc/{ppid}/comm").read().strip()
            pargv = open(f"/proc/{ppid}/cmdline").read().split("\0")
            if pcomm == "python3" and any(a.endswith(SCRIPT) for a in pargv):
                continue
        except (OSError, ValueError, IndexError):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def pump_subscribe():
    global subscribe_proc
    subscribe_proc = subprocess.Popen(
        ["pactl", "subscribe"], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True)
    try:
        for _ in subscribe_proc.stdout:
            got_event.set()
    finally:
        if subscribe_proc.poll() is None:
            subscribe_proc.kill()


def dictation_active():
    out = subprocess.run(
        ["pactl", "list", "source-outputs"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    return any(MATCH.search(b) for b in out.split("Source Output #")[1:])


def _shutdown(*_):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

reap_older_copies()
reap_orphan_subscribes()

threading.Thread(target=pump_subscribe, daemon=True).start()

state = None
last_check = 0.0
quiet_since = None
try:
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
finally:
    if subscribe_proc is not None and subscribe_proc.poll() is None:
        subscribe_proc.kill()
