# -*- coding: utf-8 -*-
"""
geminit.py — Test Gemini Live Translate (JP → VI)
Chạy: python geminit.py
Yêu cầu: pip install google-genai PyAudioWPatch numpy
"""

import os, sys, json, time, queue, asyncio, threading
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
try:
    _cfg = json.load(open(_cfg_path, encoding="utf-8"))
    GEMINI_API_KEY = _cfg.get("gemini_api_key", "")
except Exception:
    GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    print("[ERROR] Không tìm thấy gemini_api_key trong config.json")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ── Gemini ───────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    print("[ERROR] Chưa cài google-genai. Chạy: pip install google-genai")
    sys.exit(1)

# Thử translate model trước, fallback sang flash live
GEMINI_MODELS  = [
    "gemini-3.5-live-translate-preview",
]
TARGET_LANG    = "vi"   # Vietnamese
TARGET_SR      = 16000  # Hz — Gemini yêu cầu 16kHz PCM16

# ── Audio ────────────────────────────────────────────────────────────────────
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("[ERROR] Chưa cài PyAudioWPatch. Chạy: pip install PyAudioWPatch")
    sys.exit(1)

CHUNK_MS  = 100   # ms mỗi lần gửi audio tới Gemini


def list_loopback_devices():
    pa = pyaudio.PyAudio()
    devs = []
    try:
        for d in pa.get_loopback_device_info_generator():
            devs.append((int(d["index"]), d["name"]))
    finally:
        pa.terminate()
    return devs


def to_pcm16_16k(raw_bytes: bytes, src_rate: int, channels: int) -> bytes:
    """float32 WASAPI → int16 PCM 16kHz mono."""
    audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if src_rate != TARGET_SR and len(audio) > 0:
        n_out = int(round(len(audio) * TARGET_SR / src_rate))
        if n_out > 0:
            xo = np.linspace(0, 1, len(audio), endpoint=False)
            xn = np.linspace(0, 1, n_out, endpoint=False)
            audio = np.interp(xn, xo, audio).astype(np.float32)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


# ── GeminiSession ─────────────────────────────────────────────────────────────
class GeminiSession:
    """
    Chạy asyncio event loop trong 1 thread riêng.
    Nhận audio bytes từ audio_q, gửi sang Gemini.
    Nhận JP/VI text, đẩy vào jp_q / vi_q.
    """

    def __init__(self, audio_q: queue.Queue, jp_q: queue.Queue,
                 vi_q: queue.Queue, status_q: queue.Queue):
        self.audio_q  = audio_q
        self.jp_q     = jp_q
        self.vi_q     = vi_q
        self.status_q = status_q
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._session())
        except Exception as e:
            self.status_q.put(f"[Gemini] Lỗi: {e}")
        finally:
            loop.close()

    async def _session(self):
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = gtypes.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=gtypes.AudioTranscriptionConfig(),
            output_audio_transcription=gtypes.AudioTranscriptionConfig(),
            translation_config=gtypes.TranslationConfig(
                target_language_code=TARGET_LANG,
            ),
        )

        self.status_q.put("⏳ Đang kết nối Gemini Live...")
        model = GEMINI_MODELS[0]
        try:
            async with client.aio.live.connect(model=model, config=config) as session:
                self.status_q.put(f"✅ Đã kết nối ({model}) — đang nghe loa...")

                # Task gửi audio
                send_task = asyncio.create_task(self._send_audio(session))
                # Task nhận kết quả
                recv_task = asyncio.create_task(self._recv(session))

                await asyncio.gather(send_task, recv_task)
        except Exception as e:
            self.status_q.put(f"[Gemini] Session lỗi: {e}")

    async def _send_audio(self, session):
        """Lấy audio từ queue, gửi sang Gemini liên tục."""
        while not self._stop.is_set():
            try:
                pcm_bytes = self.audio_q.get(timeout=0.3)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            try:
                await session.send_realtime_input(
                    audio=gtypes.Blob(
                        data=pcm_bytes,
                        mime_type=f"audio/pcm;rate={TARGET_SR}",
                    )
                )
            except Exception as e:
                self.status_q.put(f"[Send] {e}")
                break
        # Báo hiệu kết thúc
        try:
            await session.send_realtime_input(audio_stream_end=True)
        except Exception:
            pass

    async def _recv(self, session):
        """Nhận JP transcription và VI translation từ Gemini."""
        try:
            async for msg in session.receive():
                if self._stop.is_set():
                    break
                sc = getattr(msg, "server_content", None)
                if sc is None:
                    continue

                # JP — input transcription (người nói nói gì)
                it = getattr(sc, "input_transcription", None)
                if it and getattr(it, "text", ""):
                    self.jp_q.put(it.text)

                # VI — output transcription (bản dịch audio output)
                ot = getattr(sc, "output_transcription", None)
                if ot and getattr(ot, "text", ""):
                    self.vi_q.put(ot.text)

                # VI — model text response (flash model trả text trực tiếp)
                mt = getattr(sc, "model_turn", None)
                if mt and getattr(mt, "parts", None):
                    for part in mt.parts:
                        txt = getattr(part, "text", "")
                        if txt:
                            self.vi_q.put(txt)
        except Exception as e:
            if not self._stop.is_set():
                self.status_q.put(f"[Recv] {e}")


# ── AudioLoopback ─────────────────────────────────────────────────────────────
class AudioLoopback:
    def __init__(self, audio_q: queue.Queue, device_index=None):
        self.pa           = pyaudio.PyAudio()
        self.audio_q      = audio_q
        self.device_index = device_index
        self.stream       = None
        self.src_rate     = TARGET_SR
        self.channels     = 1
        self._running     = False

    def _find_device(self):
        if self.device_index is not None:
            return self.pa.get_device_info_by_index(self.device_index)
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for d in self.pa.get_loopback_device_info_generator():
            if default_out["name"] in d["name"]:
                return d
        for d in self.pa.get_loopback_device_info_generator():
            return d
        raise RuntimeError("Không tìm thấy WASAPI loopback.")

    def start(self):
        dev = self._find_device()
        self.src_rate = int(dev["defaultSampleRate"])
        self.channels = max(1, int(dev["maxInputChannels"]))
        frames = int(self.src_rate * CHUNK_MS / 1000)
        print(f"[Audio] {dev['name']} | {self.src_rate}Hz | {self.channels}ch | {frames} frames/chunk")
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.src_rate,
            frames_per_buffer=frames,
            input=True,
            input_device_index=int(dev["index"]),
            stream_callback=self._cb,
        )
        self._running = True
        self.stream.start_stream()

    def _cb(self, in_data, frame_count, time_info, status):
        if self._running:
            pcm = to_pcm16_16k(in_data, self.src_rate, self.channels)
            self.audio_q.put(pcm)
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


# ── Tkinter UI ────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("⚡ Gemini Live Translate — JP → VI")
        root.attributes("-topmost", True)
        root.geometry("860x500+60+60")
        root.configure(bg="#0d1117")

        self.audio_q  = queue.Queue()
        self.jp_q     = queue.Queue()
        self.vi_q     = queue.Queue()
        self.status_q = queue.Queue()

        self.audio   = None
        self.gemini  = None
        self.running = False

        self._build_ui()
        root.after(100, self._poll)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Toolbar
        bar = tk.Frame(self.root, bg="#161b22")
        bar.pack(fill="x", padx=0, pady=0)

        self.btn_start = tk.Button(
            bar, text="▶ Start", width=10,
            command=self._start,
            bg="#238636", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2",
        )
        self.btn_start.pack(side="left", padx=8, pady=6)

        self.btn_stop = tk.Button(
            bar, text="■ Stop", width=10,
            command=self._stop,
            bg="#6e7681", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2", state="disabled",
        )
        self.btn_stop.pack(side="left", padx=4, pady=6)

        tk.Button(
            bar, text="🗑 Xóa", width=7,
            command=self._clear,
            bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 10),
            relief="flat", cursor="hand2",
        ).pack(side="left", padx=4, pady=6)

        tk.Button(
            bar, text="📋 Copy JP", width=10,
            command=lambda: self._copy(self.txt_jp),
            bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 10),
            relief="flat", cursor="hand2",
        ).pack(side="left", padx=2, pady=6)

        tk.Button(
            bar, text="📋 Copy VI", width=10,
            command=lambda: self._copy(self.txt_vi),
            bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 10),
            relief="flat", cursor="hand2",
        ).pack(side="left", padx=2, pady=6)

        # Device selector
        dev_row = tk.Frame(self.root, bg="#0d1117")
        dev_row.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(dev_row, text="🔊 Thiết bị:", bg="#0d1117",
                 fg="#8b949e", font=("Segoe UI", 9)).pack(side="left")
        self.devices  = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or ["(Không tìm thấy loopback)"]
        self.dev_var  = tk.StringVar()
        self.dev_combo = ttk.Combobox(
            dev_row, textvariable=self.dev_var,
            values=dev_names, state="readonly", width=55,
        )
        self.dev_combo.pack(side="left", padx=6)
        default = next(
            (i for i, n in enumerate(dev_names) if "Headphone" in n or "Realtek" in n or "Speaker" in n), 0
        )
        self.dev_combo.current(default)

        # Status bar
        self.lbl_status = tk.Label(
            self.root, text="Chưa kết nối",
            bg="#0d1117", fg="#8b949e",
            font=("Segoe UI", 9), anchor="w",
        )
        self.lbl_status.pack(fill="x", padx=12, pady=(4, 2))

        # Hai panel JP / VI song song
        panels = tk.Frame(self.root, bg="#0d1117")
        panels.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # JP panel
        jp_frame = tk.Frame(panels, bg="#0d1117")
        jp_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        tk.Label(jp_frame, text="🇯🇵 Tiếng Nhật (transcript)",
                 bg="#0d1117", fg="#58a6ff",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.txt_jp = scrolledtext.ScrolledText(
            jp_frame, wrap="word",
            font=("Yu Gothic UI", 14),
            bg="#161b22", fg="#e6edf3",
            insertbackground="#e6edf3",
            relief="flat", borderwidth=1,
        )
        self.txt_jp.pack(fill="both", expand=True)

        # Divider
        tk.Frame(panels, bg="#30363d", width=1).pack(side="left", fill="y")

        # VI panel
        vi_frame = tk.Frame(panels, bg="#0d1117")
        vi_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        tk.Label(vi_frame, text="🇻🇳 Tiếng Việt (dịch)",
                 bg="#0d1117", fg="#3fb950",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.txt_vi = scrolledtext.ScrolledText(
            vi_frame, wrap="word",
            font=("Segoe UI", 14),
            bg="#161b22", fg="#aff3c3",
            insertbackground="#aff3c3",
            relief="flat", borderwidth=1,
        )
        self.txt_vi.pack(fill="both", expand=True)

    # ── Control ───────────────────────────────────────────────────────────────
    def _selected_device_index(self):
        sel = self.dev_var.get()
        for idx, name in self.devices:
            if name == sel:
                return idx
        return None

    def _start(self):
        if self.running:
            return
        self.running = True
        self.btn_start.config(state="disabled", bg="#6e7681")
        self.btn_stop.config(state="normal", bg="#da3633")
        self.dev_combo.config(state="disabled")

        # Khởi động Gemini session
        self.gemini = GeminiSession(self.audio_q, self.jp_q, self.vi_q, self.status_q)
        self.gemini.start()

        # Khởi động audio loopback
        self.audio = AudioLoopback(self.audio_q, device_index=self._selected_device_index())
        try:
            self.audio.start()
        except Exception as e:
            self._set_status(f"❌ Lỗi audio: {e}")
            self._stop()

    def _stop(self):
        if not self.running:
            return
        self.running = False
        self.btn_start.config(state="normal", bg="#238636")
        self.btn_stop.config(state="disabled", bg="#6e7681")
        self.dev_combo.config(state="readonly")

        if self.gemini:
            self.gemini.stop()
            self.gemini = None
        if self.audio:
            self.audio.terminate()
            self.audio = None

        self._set_status("Đã dừng")

    def _clear(self):
        self.txt_jp.delete("1.0", "end")
        self.txt_vi.delete("1.0", "end")

    def _copy(self, widget: scrolledtext.ScrolledText):
        content = widget.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._flash_status("✅ Đã copy!")

    def _set_status(self, msg: str, color="#8b949e"):
        self.lbl_status.config(text=msg, fg=color)

    def _flash_status(self, msg: str):
        old = self.lbl_status.cget("text")
        self._set_status(msg, "#3fb950")
        self.root.after(2000, lambda: self._set_status(old))

    # ── Poll queues mỗi 100ms ─────────────────────────────────────────────────
    def _poll(self):
        # Status từ Gemini session
        while not self.status_q.empty():
            msg = self.status_q.get()
            color = "#3fb950" if "✅" in msg else "#f85149" if "❌" in msg or "lỗi" in msg.lower() else "#8b949e"
            self._set_status(msg, color)

        # JP text
        while not self.jp_q.empty():
            text = self.jp_q.get().strip()
            if text:
                self.txt_jp.insert("end", text)
                self.txt_jp.see("end")

        # VI text
        while not self.vi_q.empty():
            text = self.vi_q.get().strip()
            if text:
                self.txt_vi.insert("end", text)
                self.txt_vi.see("end")

        self.root.after(100, self._poll)

    def _on_close(self):
        self._stop()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
