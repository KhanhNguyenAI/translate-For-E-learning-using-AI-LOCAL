# -*- coding: utf-8 -*-
"""
AudioMic class + list_mic_devices().
"""

import sys

from config import TARGET_SR

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("Thiếu PyAudioWPatch. Chạy: pip install PyAudioWPatch")
    sys.exit(1)


def list_mic_devices():
    pa = pyaudio.PyAudio()
    devices = []
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (info.get("hostApi") == wasapi["index"]
                    and int(info.get("maxInputChannels", 0)) > 0
                    and "loopback" not in info.get("name", "").lower()):
                devices.append((int(info["index"]), info["name"]))
    finally:
        pa.terminate()
    return devices


class AudioMic:
    """Thu âm từ microphone (WASAPI input, không phải loopback)."""

    def __init__(self, frame_queue, device_index=None):
        self.pa = pyaudio.PyAudio()
        self.frame_queue = frame_queue
        self.device_index = device_index
        self.stream = None
        self.device_rate = TARGET_SR
        self.channels = 1
        self._running = False

    def _find_device(self):
        if self.device_index is not None:
            return self.pa.get_device_info_by_index(self.device_index)
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_in = self.pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        return default_in

    def start(self):
        dev = self._find_device()
        self.device_rate = int(dev["defaultSampleRate"])
        self.channels = max(1, int(dev["maxInputChannels"]))
        print(f"[Mic] {dev['name']} | {self.device_rate}Hz | {self.channels}ch")
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.device_rate,
            frames_per_buffer=1024,
            input=True,
            input_device_index=int(dev["index"]),
            stream_callback=self._callback,
        )
        self._running = True
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        if self._running:
            self.frame_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def stop(self):
        self._running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def terminate(self):
        self.stop()
        self.pa.terminate()
