# -*- coding: utf-8 -*-
"""
ReazonSpeech transcriber thread — fast Japanese-specialized STT (k2-asr).
"""

import time
import queue
import threading

import numpy as np

from config import (
    TARGET_SR, CHUNK_SEC, OVERLAP_SEC, RMS_THRESHOLD, HALLUCINATIONS,
    VAD_SILENCE_HANG, VAD_MIN_SPEECH, VAD_MAX_SEG,
)
from audio.loopback import to_mono_16k


class ReazonSpeechTranscriber(threading.Thread):
    def __init__(self, frame_queue, text_queue, status_queue, get_params,
                 use_diarization=None, get_src_lang=None, chunk_sec=None):
        super().__init__(daemon=True)
        self.frame_queue   = frame_queue
        self.text_queue    = text_queue
        self.status_queue  = status_queue
        self.get_params    = get_params
        self._get_src_lang = get_src_lang or (lambda: "ja")
        self._auto         = (chunk_sec == "auto")
        self._chunk_sec    = CHUNK_SEC if self._auto else (chunk_sec or CHUNK_SEC)
        self.model = None
        self._stop = threading.Event()
        self.model_ready = threading.Event()

    def load_model(self):
        self.status_queue.put("Loading ReazonSpeech (k2, JA)...")
        try:
            from huggingface_hub import login
            from config import HF_TOKEN
            if HF_TOKEN:
                login(token=HF_TOKEN)
            from reazonspeech.k2.asr import load_model
            self.model = load_model(language="ja")
            self.model_ready.set()
            self.status_queue.put("ReazonSpeech ready. Listening...")
        except Exception as e:
            print(f"[ReazonSpeech] Load error: {e}")
            self.status_queue.put(f"❌ ReazonSpeech load failed: {e}")

    def run(self):
        self.load_model()
        if self._auto:
            self._run_auto()
        else:
            self._run_fixed()

    def _run_fixed(self):
        buf = np.zeros(0, dtype=np.float32)
        overlap = np.zeros(0, dtype=np.float32)
        chunk_samples = int(self._chunk_sec * TARGET_SR)
        last_vol_t = time.time()

        while not self._stop.is_set():
            try:
                raw = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            device_rate, channels = self.get_params()
            mono = to_mono_16k(raw, device_rate, channels)
            buf = np.concatenate([buf, mono])

            now = time.time()
            if now - last_vol_t > 1.5 and len(buf) > 0:
                rms = float(np.sqrt(np.mean(buf**2)))
                filled = int(min(rms * 300, 20))
                bar = "█" * filled + "░" * (20 - filled)
                self.status_queue.put(f"Listening...  [{bar}]  {rms:.4f}")
                last_vol_t = now

            if len(buf) >= chunk_samples:
                audio_in = np.concatenate([overlap, buf])
                overlap = buf[-int(OVERLAP_SEC * TARGET_SR):].copy()
                buf = np.zeros(0, dtype=np.float32)
                self._transcribe(audio_in)

    def _run_auto(self):
        """Endpoint mode: gom audio đến khi người nói ngừng (im lặng) thì chốt câu."""
        speech = np.zeros(0, dtype=np.float32)
        in_speech = False
        silence_run = 0.0
        voiced_samples = 0
        last_vol_t = time.time()
        min_samples = int(VAD_MIN_SPEECH * TARGET_SR)
        max_samples = int(VAD_MAX_SEG * TARGET_SR)

        while not self._stop.is_set():
            try:
                raw = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            device_rate, channels = self.get_params()
            mono = to_mono_16k(raw, device_rate, channels)
            if len(mono) == 0:
                continue
            block_dur = len(mono) / TARGET_SR
            block_rms = float(np.sqrt(np.mean(mono ** 2)))

            now = time.time()
            if now - last_vol_t > 1.0:
                filled = int(min(block_rms * 300, 20))
                bar = "█" * filled + "░" * (20 - filled)
                state = "speech" if in_speech else "silence"
                self.status_queue.put(f"Listening (auto/{state})...  [{bar}]  {block_rms:.4f}")
                last_vol_t = now

            if block_rms >= RMS_THRESHOLD:
                in_speech = True
                silence_run = 0.0
                speech = np.concatenate([speech, mono])
                voiced_samples += len(mono)
            elif in_speech:
                speech = np.concatenate([speech, mono])
                silence_run += block_dur
                if silence_run >= VAD_SILENCE_HANG:
                    if voiced_samples >= min_samples:   # real sentence — flush
                        self._transcribe(speech)
                    # else: just a noise blip — discard
                    speech = np.zeros(0, dtype=np.float32)
                    in_speech = False
                    silence_run = 0.0
                    voiced_samples = 0
                    continue

            if len(speech) >= max_samples:
                if voiced_samples >= min_samples:
                    self._transcribe(speech)
                speech = np.zeros(0, dtype=np.float32)
                in_speech = False
                silence_run = 0.0
                voiced_samples = 0

    def _transcribe(self, audio):
        try:
            from tts.engine import tts_speaking, keep_recording_during_tts
            if tts_speaking.is_set() and not keep_recording_during_tts.is_set():
                return
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < RMS_THRESHOLD:
                return
            if self.model is None:
                return

            from reazonspeech.k2.asr import transcribe, audio_from_numpy, TranscribeConfig

            audio_data = audio_from_numpy(audio, TARGET_SR)
            result = transcribe(self.model, audio_data, TranscribeConfig(verbose=False))

            full_text = result.text.strip()
            if not full_text:
                return

            cur_lang = self._get_src_lang()
            stripped = full_text.replace("。", "").replace("、", "").replace(" ", "").strip()
            hall_set = HALLUCINATIONS.get(cur_lang, set())
            if stripped in hall_set:
                return
            from tts.engine import is_recent_tts
            if is_recent_tts(full_text):   # echo of our own TTS — drop
                return

            self.text_queue.put({"text": full_text, "speaker": None, "color": "#eee"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_queue.put(f"ReazonSpeech error: {e}")

    def stop(self):
        self._stop.set()
