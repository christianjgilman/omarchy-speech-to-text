// Dictation Pill — click-through overlay shown while a dictation app captures
// the microphone. Renders on the focused monitor; bars track the real mic
// level (mic-level.py). Detection and race handling live in dictation-watch.py.
//
// Which apps trigger the pill: edit `dictationMatch` below (regex matched
// against each capture stream's pactl block, e.g. "wispr|whisper").

import QtQuick
import Quickshell
import Quickshell.Io
import Quickshell.Wayland
import Quickshell.Hyprland

Item {
    id: root

    // Apps to watch for, matched against each capture stream's pactl block.
    readonly property string dictationMatch: "wispr|voxtype|dictationd"
    // This plugin's own directory (scripts live next to this file).
    readonly property string pluginDir: Qt.resolvedUrl(".").toString().replace(/^file:\/\//, "").replace(/\/$/, "")

    property bool recording: false
    property real level: 0
    property string focusedName: Hyprland.focusedMonitor ? Hyprland.focusedMonitor.name : ""

    // --- detection -----------------------------------------------------------
    // One long-lived watcher process; prints "1"/"0" on state changes only.
    // All race/debounce handling lives in dictation-watch.py.
    Process {
        id: watcher
        running: true
        command: ["python3", root.pluginDir + "/dictation-watch.py", root.dictationMatch]
        stdout: SplitParser {
            onRead: function(line) {
                root.recording = line.trim() === "1"
                if (!root.recording) fadeHold.restart()
            }
        }
        onExited: restartWatch.restart()
    }
    Timer {
        id: restartWatch
        interval: 2000
        onTriggered: watcher.running = true
    }

    // Holds the window mapped through the fade-out.
    Timer {
        id: fadeHold
        interval: 120
    }

    onRecordingChanged: if (!recording) level = 0

    // --- live mic level --------------------------------------------------------
    // Only runs while the pill is showing: no mic activity when idle.
    Process {
        id: levelProcess
        running: root.recording
        command: ["python3", root.pluginDir + "/mic-level.py"]
        stdout: SplitParser {
            onRead: function(line) {
                const v = parseFloat(line)
                if (!isNaN(v)) root.level = v
            }
        }
    }

    // --- overlay -------------------------------------------------------------
    PanelWindow {
        id: win

        screen: Quickshell.screens.find(s => s.name === root.focusedName) ?? Quickshell.screens[0]
        anchors {
            left: true
            right: true
            bottom: true
        }
        margins {
            bottom: 22
        }
        implicitHeight: 40
        color: "transparent"
        exclusionMode: ExclusionMode.Ignore
        // Mapped only while shown (or fading out); no input region anywhere.
        visible: root.recording || fadeHold.running

        WlrLayershell.namespace: "dictation-pill"
        WlrLayershell.layer: WlrLayer.Overlay
        WlrLayershell.keyboardFocus: WlrKeyboardFocus.None

        // Empty region => the surface never receives pointer input:
        // clicks, hover and focus all pass through to windows below.
        mask: Region {
            width: 0
            height: 0
        }

        Rectangle {
            id: pill
            anchors.centerIn: parent
            width: row.implicitWidth + 22
            height: 26
            radius: 13
            color: "#E6111519"
            border.color: "#2EFFFFFF"
            border.width: 1

            opacity: root.recording ? 1 : 0
            scale: root.recording ? 1 : 0.94

            // Identical in and out: same duration, same easing, both properties.
            Behavior on opacity {
                NumberAnimation {
                    duration: 90
                    easing.type: Easing.OutQuad
                }
            }
            Behavior on scale {
                NumberAnimation {
                    duration: 90
                    easing.type: Easing.OutQuad
                }
            }

            Row {
                id: row
                anchors.centerIn: parent
                spacing: 8

                // Live waveform bars — driven by the real mic level.
                Row {
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 2.5
                    Repeater {
                        model: 5
                        Rectangle {
                            width: 2.5
                            radius: 1.25
                            color: "#EAEFFF"
                            anchors.verticalCenter: parent.verticalCenter

                            // Per-bar gain and spring speed give an organic wave.
                            readonly property real gain: [1.0, 0.75, 1.25, 0.9, 0.65][index]
                            readonly property int dur: [45, 60, 75, 90, 105][index]
                            readonly property real minH: 3
                            readonly property real maxH: 13

                            height: minH
                            Behavior on height {
                                NumberAnimation {
                                    duration: dur
                                    easing.type: Easing.OutQuad
                                }
                            }

                            Connections {
                                target: root
                                function onLevelChanged() {
                                    const driven = Math.min(1, root.level * gain)
                                    height = minH + (maxH - minH) * driven
                                }
                            }
                        }
                    }
                }

                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: "Listening"
                    color: "#EAEFFF"
                    font.family: "Inter"
                    font.pointSize: 8
                    font.weight: Font.DemiBold
                }
            }
        }
    }
}
