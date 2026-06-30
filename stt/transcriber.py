# -*- coding: utf-8 -*-
"""
Transcriber thread — runs Whisper model in background.
"""

import time
import queue
import threading

import numpy as np
from faster_whisper import WhisperModel

from config import (
    MODEL_SIZE, DEVICE, COMPUTE_TYPE, TARGET_SR,
    CHUNK_SEC, OVERLAP_SEC, RMS_THRESHOLD, HALLUCINATIONS,
    VAD_SILENCE_HANG, VAD_MIN_SPEECH, VAD_MAX_SEG,
)
from audio.loopback import to_mono_16k
import stt.diarization as _diar_mod
from stt.diarization import diarize_audio, assign_speaker_color


class Transcriber(threading.Thread):
    def __init__(self, frame_queue, text_queue, status_queue, get_params,
                 use_diarization=None, get_src_lang=None, chunk_sec=None):
        super().__init__(daemon=True)
        self.frame_queue      = frame_queue
        self.text_queue       = text_queue
        self.status_queue     = status_queue
        self.get_params       = get_params
        self._use_diarization = use_diarization  # tk.BooleanVar
        self._get_src_lang    = get_src_lang or (lambda: "ja")
        self._auto            = (chunk_sec == "auto")
        self._chunk_sec       = CHUNK_SEC if self._auto else (chunk_sec or CHUNK_SEC)
        self.model = None
        self._stop = threading.Event()
        self.model_ready = threading.Event()

    def load_model(self):
        self.status_queue.put(f"Loading model '{MODEL_SIZE}' ({DEVICE})...")
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            print(f"[Whisper] GPU error: {e}")
            self.status_queue.put(f"GPU error ({e}); switching to CPU...")
            try:
                self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            except Exception as e2:
                print(f"[Whisper] CPU error: {e2}")
                self.status_queue.put(f"❌ Whisper load failed: {e2}")
                return
        self.model_ready.set()
        if _diar_mod.HAS_DIARIZATION:
            self.status_queue.put("Ready (Diarization ON). Listening...")
        else:
            self.status_queue.put("Ready. Listening...")

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
            diar_on = (self._use_diarization is None or self._use_diarization.get())
            diar_future = []
            if _diar_mod.HAS_DIARIZATION and diar_on:
                t = threading.Thread(
                    target=lambda: diar_future.append(diarize_audio(audio)),
                    daemon=True,
                )
                t.start()

            cur_lang = self._get_src_lang()
            segments, info = self.model.transcribe(
                audio, language=cur_lang, beam_size=2,
                vad_filter=True, condition_on_previous_text=False,
                no_speech_threshold=0.5,
                word_timestamps=True,
            )
            seg_list = list(segments)

            if info.language_probability < 0.5:
                return

            all_words = []
            for seg in seg_list:
                for w in (seg.words or []):
                    all_words.append(w)

            full_text = "".join(s.text for s in seg_list).strip()
            if not full_text:
                return
            stripped = full_text.replace("。","").replace("、","").replace(" ","").strip()
            hall_set = HALLUCINATIONS.get(cur_lang, set())
            if stripped in hall_set:
                return
            from tts.engine import is_recent_tts
            if is_recent_tts(full_text):   # echo of our own TTS — drop
                return

            if _diar_mod.HAS_DIARIZATION and diar_on:
                t.join(timeout=8.0)
                turns = diar_future[0] if diar_future else []

                if turns and all_words:
                    # Khớp từng word -> diarization turn -> gom theo speaker liên tiếp
                    def find_speaker(t_mid):
                        for start, end, spk in turns:
                            if start <= t_mid <= end:
                                return spk
                        # Nếu không khớp -> lấy turn gần nhất
                        return min(turns, key=lambda x: abs((x[0]+x[1])/2 - t_mid))[2]

                    # Gom words liên tiếp cùng speaker thành 1 segment
                    grouped = []
                    for w in all_words:
                        t_mid = (w.start + w.end) / 2
                        spk   = find_speaker(t_mid)
                        if grouped and grouped[-1][0] == spk:
                            grouped[-1][1].append(w.word)
                        else:
                            grouped.append([spk, [w.word]])

                    for spk, words_list in grouped:
                        seg_text = "".join(words_list).strip()
                        if not seg_text:
                            continue
                        color, label = assign_speaker_color(spk)
                        self.text_queue.put({"text": seg_text, "speaker": label, "color": color})
                    return

                elif turns:
                    # Không có word timestamps -> fallback dominant speaker
                    from collections import Counter
                    spk_times = Counter()
                    for s, e, spk in turns:
                        spk_times[spk] += (e - s)
                    dominant = spk_times.most_common(1)[0][0]
                    color, label = assign_speaker_color(dominant)
                    self.text_queue.put({"text": full_text, "speaker": label, "color": color})
                    return

            self.text_queue.put({"text": full_text, "speaker": None, "color": "#eee"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_queue.put(f"Recognition error: {e}")

    def stop(self):
        self._stop.set()
