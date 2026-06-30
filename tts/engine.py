# -*- coding: utf-8 -*-
"""
TTSThread, TTS playback, voice listing (Edge TTS + pygame).
"""

import os
import time
import queue
import threading

TTS_DEFAULT_VOICE = {
    "vi": "vi-VN-NamMinhNeural",
    "ja": "ja-JP-KeitaNeural",
    "en": "en-US-AndrewNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "my": "my-MM-NilarNeural",
}

LANG_TO_LOCALE = {
    "vi": "vi-VN", "ja": "ja-JP", "en": "en-US", "zh": "zh-CN", "my": "my-MM",
}

TTS_SPEED_OPTIONS = [
    ("1x",   "+0%"),
    ("1.5x", "+50%"),
    ("2x",   "+100%"),
    ("2.5x", "+150%"),
    ("3x",   "+200%"),
]

HAS_TTS = False
try:
    import edge_tts, asyncio, io
    HAS_TTS = True
except ImportError:
    print("[TTS] edge-tts chưa cài — pip install edge-tts")

HAS_PYGAME = False
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    print("[TTS] pygame chưa cài — pip install pygame")

_all_tts_voices: dict[str, list[dict]] = {}

# Set while TTS audio is actually playing.
tts_speaking = threading.Event()

# When set, transcribers DO NOT pause during TTS (rely on language filter +
# the recent-TTS-text dedup below to reject the spoken-back translation).
keep_recording_during_tts = threading.Event()

# ── Recent-TTS-text registry (dedup safety net) ───────────────────────
_recent_tts_lock = threading.Lock()
_recent_tts: list[tuple[str, float]] = []   # (normalized_text, expiry_ts)
_RECENT_TTS_TTL = 8.0


def _norm_tts(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def note_tts_text(text: str):
    """Remember a phrase we just spoke, so STT can drop it if captured back."""
    n = _norm_tts(text)
    if not n:
        return
    with _recent_tts_lock:
        _recent_tts.append((n, time.time() + _RECENT_TTS_TTL))


def is_recent_tts(text: str) -> bool:
    """True if `text` matches something TTS spoke recently (echo of our own voice)."""
    n = _norm_tts(text)
    if not n:
        return False
    now = time.time()
    with _recent_tts_lock:
        _recent_tts[:] = [(t, e) for (t, e) in _recent_tts if e > now]
        for t, _ in _recent_tts:
            if n in t or t in n:
                return True
    return False


def _load_all_tts_voices():
    """Load voice list từ Edge TTS, cache theo locale."""
    global _all_tts_voices
    if not HAS_TTS:
        return
    try:
        loop = asyncio.new_event_loop()
        voices = loop.run_until_complete(edge_tts.list_voices())
        loop.close()
        for v in voices:
            locale = v["Locale"]
            if locale not in _all_tts_voices:
                _all_tts_voices[locale] = []
            _all_tts_voices[locale].append(v)
    except Exception as e:
        print(f"[TTS] Failed to load voices: {e}")


def get_voices_for_lang(lang_code: str) -> list[dict]:
    locale = LANG_TO_LOCALE.get(lang_code, "")
    return _all_tts_voices.get(locale, [])


def list_output_devices() -> list[tuple[str, str]]:
    """Trả về list (device_name, display_name) cho audio output."""
    if not HAS_PYGAME:
        return []
    try:
        import pygame._sdl2.audio as sdl2_audio
        if not pygame.get_init():
            pygame.init()
        names = sdl2_audio.get_audio_device_names(False)
        return [(n, n) for n in names]
    except Exception:
        return []


def _play_mp3_on_device(filepath: str, device_name: str | None = None):
    """Phát MP3 bằng pygame.mixer trên device chỉ định."""
    if not HAS_PYGAME:
        return
    try:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        if device_name:
            pygame.mixer.init(devicename=device_name)
        else:
            pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    except Exception as e:
        print(f"[TTS] Playback error: {e}")
    finally:
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass


class TTSThread(threading.Thread):
    """Đọc bản dịch bằng Edge TTS (Microsoft Neural Voices)."""

    def __init__(self, tts_queue, get_tgt_lang=None, get_output_device=None,
                 get_voice=None, get_rate=None):
        super().__init__(daemon=True)
        self.tts_queue = tts_queue
        self._get_tgt_lang = get_tgt_lang or (lambda: "vi")
        self._get_output_device = get_output_device or (lambda: None)
        self._get_voice = get_voice or (lambda: None)
        self._get_rate = get_rate or (lambda: "+100%")
        self._stop = threading.Event()
        self._loop = None

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        while not self._stop.is_set():
            try:
                text = self.tts_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if not text or not text.strip():
                continue
            try:
                self._loop.run_until_complete(self._speak(text))
            except Exception as e:
                print(f"[TTS] Error: {e}")
        self._loop.close()

    async def _speak(self, text: str):
        import tempfile
        lang = self._get_tgt_lang()
        voice = self._get_voice() or TTS_DEFAULT_VOICE.get(lang, "en-US-AriaNeural")
        rate = self._get_rate()
        comm = edge_tts.Communicate(text, voice, rate=rate)
        buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        if buf.tell() == 0:
            return
        buf.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(buf.read())
            tmp_path = f.name
        try:
            note_tts_text(text)
            tts_speaking.set()
            _play_mp3_on_device(tmp_path, self._get_output_device())
        finally:
            tts_speaking.clear()
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def stop(self):
        self._stop.set()
