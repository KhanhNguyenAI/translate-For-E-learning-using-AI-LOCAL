# -*- coding: utf-8 -*-
"""
AudioLoopback class + list_loopback_devices() + to_mono_16k helper.
"""

import sys
import numpy as np

from config import TARGET_SR

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("Thiếu PyAudioWPatch. Chạy: pip install PyAudioWPatch")
    sys.exit(1)


def list_loopback_devices():
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for dev in pa.get_loopback_device_info_generator():
            devices.append((int(dev["index"]), dev["name"]))
    finally:
        pa.terminate()
    return devices


def to_mono_16k(raw_bytes, device_rate, channels):
    audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if device_rate != TARGET_SR and len(audio) > 0:
        n_out = int(round(len(audio) * TARGET_SR / device_rate))
        if n_out > 0:
            x_old = np.linspace(0, 1, len(audio), endpoint=False)
            x_new = np.linspace(0, 1, n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


class AudioLoopback:
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
        default_out = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for dev in self.pa.get_loopback_device_info_generator():
            if default_out["name"] in dev["name"]:
                return dev
        for dev in self.pa.get_loopback_device_info_generator():
            return dev
        raise RuntimeError("Không tìm thấy WASAPI loopback nào.")

    def start(self):
        dev = self._find_device()
        self.device_rate = int(dev["defaultSampleRate"])
        self.channels = max(1, int(dev["maxInputChannels"]))
        print(f"[Audio] {dev['name']} | {self.device_rate}Hz | {self.channels}ch")
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
