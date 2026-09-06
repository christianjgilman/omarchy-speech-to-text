# Emits the live peak level (0..1) of the default mic, one number per ~50ms.
# Runs only while the dictation pill is showing; spawned by pill.qml.
import array
import subprocess
import sys

RATE = 16000
CHUNK = 800  # 50 ms of s16le mono

proc = subprocess.Popen(
    ["pacat", "--record", "--raw", "--rate", str(RATE),
     "--channels=1", "--format=s16le", "--latency-msec=50"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

try:
    while True:
        buf = proc.stdout.read(CHUNK * 2)
        if not buf or len(buf) < CHUNK * 2:
            break
        samples = array.array("h")
        samples.frombytes(buf)
        peak = max(abs(s) for s in samples) / 32768.0
        print(f"{min(1.0, peak * 1.8):.3f}", flush=True)
except KeyboardInterrupt:
    pass
finally:
    proc.terminate()
