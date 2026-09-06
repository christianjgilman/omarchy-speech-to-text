// SpeechService — supervises the dictationd engine (python daemon + OpenAI
// compatible STT endpoint on 127.0.0.1:8765). Runs `setup.sh daemon`, which
// idempotently installs the venv/models on first run, then execs the daemon.
// If an externally managed dictationd is already listening, this stands down.

import QtQuick
import Quickshell
import Quickshell.Io

Item {
    id: root

    property string pluginDir: Qt.resolvedUrl(".").toString().replace(/^file:\/\//, "").replace(/\/$/, "")
    property bool daemonManaged: false
    property int restarts: 0

    function socketPath() {
        return (Quickshell.env("XDG_RUNTIME_DIR") || "/run/user/1000") + "/dictationd.sock"
    }

    // Probe: is a dictationd daemon actually answering? (A stale socket file
    // from an unclean kill passes `test -S`, so ask the daemon itself.)
    Process {
        id: probe
        command: ["bash", "-c",
                  "command -v dictationctl >/dev/null 2>&1 && dictationctl status 2>/dev/null || echo free"]
        stdout: StdioCollector {
            waitForEnd: true
            onStreamFinished: {
                if (text.trim() === "free") {
                    root.daemonManaged = true
                    daemon.running = true
                } else {
                    console.log("speech-to-text: externally managed dictationd found; standing down")
                }
            }
        }
    }

    Process {
        id: daemon
        command: ["bash", root.pluginDir + "/setup.sh", "daemon"]
        stdout: SplitParser {
            onRead: function(line) { console.log("speech-to-text:", line) }
        }
        stderr: SplitParser {
            onRead: function(line) { console.log("speech-to-text:", line) }
        }
        onExited: {
            if (root.daemonManaged) restartDaemon.restart()
        }
    }
    Timer {
        id: restartDaemon
        interval: 3000
        onTriggered: {
            if (root.daemonManaged) daemon.running = true
        }
    }

    Component.onCompleted: probe.running = true
}
