# -*- coding: utf-8 -*-
"""
Class App — main tkinter UI, poll loops, orchestration.
"""

import os
import json
import time
import queue
import ctypes
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

from config import (
    SUPPORTED_LANGS, DEFAULT_FONT_SIZE, HF_TOKEN,
)
from audio import AudioLoopback, AudioMic, list_loopback_devices, list_mic_devices
from stt import (
    Transcriber, load_diarization_pipeline, HAS_DIARIZATION, _speaker_registry,
)
from translation import TERMS_PATH, _custom_terms, _build_terms_hint
from translation.qwen import TranslatorThread
from tts import (
    TTSThread, list_output_devices, get_voices_for_lang,
    TTS_DEFAULT_VOICE, TTS_SPEED_OPTIONS,
    HAS_TTS, _load_all_tts_voices, _play_mp3_on_device,
)
from ai import (
    open_or_focus_copilot, find_copilot_hwnd, click_copilot_input,
    open_or_focus_claude, find_claude_hwnd, click_claude_input,
    open_or_focus_chatgpt, find_chatgpt_hwnd, click_chatgpt_input,
    AI_OPTIONS, _AI_REGISTRY,
)

# Lazy imports for TTS preview
try:
    import edge_tts, asyncio, io
except ImportError:
    pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("Interview STT — Multi-Language")
        root.attributes("-topmost", True)
        root.geometry("960x520+40+40")

        self.frame_queue    = queue.Queue()
        self.text_queue     = queue.Queue()
        self.status_queue   = queue.Queue()
        self.jp_trans_queue = queue.Queue()   # src text queue (kept name for compat)
        self.vi_queue       = queue.Queue()   # tgt text queue
        self.err_queue      = queue.Queue()

        self.audio = None
        self.transcriber = None
        self.translator_thread = None
        self.tts_thread = None
        self.tts_queue = queue.Queue()
        self.running = False
        self._last_speaker = None
        self._font_size = DEFAULT_FONT_SIZE
        self._dual_mode = True
        self._translate_on = True
        self._tts_on = False
        self._seg_counter = 0
        self._pending_vi = {}
        self._vi_next_id = 0

        # ── Language dropdown options ─────────────────────────────────
        self._lang_codes = list(SUPPORTED_LANGS.keys())
        lang_display = [f"{SUPPORTED_LANGS[c]['flag']} {SUPPORTED_LANGS[c]['name']}"
                        for c in self._lang_codes]

        # ── Toolbar ────────────────────────────────────────────────────
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=(6, 2))

        # - Group 1: Language pair -
        self._src_lang_var = tk.StringVar()
        self._src_combo = ttk.Combobox(
            top, textvariable=self._src_lang_var,
            values=lang_display, state="readonly", width=13,
        )
        self._src_combo.current(0)  # ja
        self._src_combo.pack(side="left", padx=(0, 2))
        self._src_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        tk.Label(top, text="→", font=("Segoe UI", 10, "bold"),
                 fg="#888").pack(side="left")

        self._tgt_lang_var = tk.StringVar()
        self._tgt_combo = ttk.Combobox(
            top, textvariable=self._tgt_lang_var,
            values=lang_display, state="readonly", width=13,
        )
        self._tgt_combo.current(self._lang_codes.index("vi"))
        self._tgt_combo.pack(side="left", padx=(2, 4))
        self._tgt_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        # - Start button -
        self.btn = tk.Button(top, text="▶ Start", width=8, command=self.toggle,
                             bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"))
        self.btn.pack(side="left", padx=(0, 2))

        # - Separator -
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        # - Group 2: Text actions -
        tk.Button(top, text="\U0001f5d1", width=3,
                  command=self.clear).pack(side="left", padx=1)
        tk.Button(top, text="\U0001f4cbL", width=3,
                  command=self.copy_jp).pack(side="left", padx=1)
        tk.Button(top, text="\U0001f4cbR", width=3,
                  command=self.copy_vi).pack(side="left", padx=1)

        # - Separator -
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        # - Group 3: AI — click gửi, ▾ chọn model -
        self._ai_choice = "Copilot"

        ai_frame = tk.Frame(top)
        ai_frame.pack(side="left", padx=2)

        self._ai_send_btn = tk.Button(
            ai_frame, text=f"\U0001f916 Copilot", width=10,
            command=self._send_to_ai,
            bg="#0078d4", fg="white", font=("Segoe UI", 9, "bold"),
        )
        self._ai_send_btn.pack(side="left")

        self._ai_drop = tk.Menubutton(
            ai_frame, text="▾", width=2,
            bg="#0078d4", fg="white", font=("Segoe UI", 9, "bold"),
            relief="raised", activebackground="#005a9e", activeforeground="white",
        )
        self._ai_menu = tk.Menu(self._ai_drop, tearoff=0,
                                 bg="#21262d", fg="#c9d1d9",
                                 activebackground="#30363d", activeforeground="white",
                                 font=("Segoe UI", 10))
        for ai_name, ai_cfg in AI_OPTIONS.items():
            self._ai_menu.add_command(
                label=f"{ai_cfg['icon']}  {ai_name}",
                command=lambda n=ai_name, c=ai_cfg: self._select_ai(n, c),
            )
        self._ai_drop.config(menu=self._ai_menu)
        self._ai_drop.pack(side="left")

        # - Separator -
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        # - Group 4: Toggles & Settings -
        self._use_diarization = tk.BooleanVar(value=True)
        tk.Checkbutton(
            top, text="\U0001f465", variable=self._use_diarization,
            font=("Segoe UI", 9), fg="#aaa", selectcolor="#1a1a2e",
        ).pack(side="left", padx=1)

        self._use_translate = tk.BooleanVar(value=True)
        self._use_translate.trace_add("write", self._on_translate_toggle)
        tk.Checkbutton(
            top, text="\U0001f310", variable=self._use_translate,
            font=("Segoe UI", 9, "bold"), fg="#64b5f6", selectcolor="#1a1a2e",
        ).pack(side="left", padx=1)

        self._use_tts = tk.BooleanVar(value=False)
        self._use_tts.trace_add("write", self._on_tts_toggle)
        tts_state = "normal" if HAS_TTS else "disabled"
        tk.Checkbutton(
            top, text="\U0001f508", variable=self._use_tts,
            font=("Segoe UI", 9, "bold"), fg="#ffb74d", selectcolor="#1a1a2e",
            state=tts_state,
        ).pack(side="left", padx=1)

        self.dual_btn = tk.Button(
            top, text="\U0001f4d6", width=2, command=self._toggle_dual,
            bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9),
        )
        self.dual_btn.pack(side="left", padx=1)

        tk.Button(top, text="A-", width=2, command=self._font_down,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side="left", padx=1)
        tk.Button(top, text="A+", width=2, command=self._font_up,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side="left", padx=1)

        tk.Button(top, text="⚙", width=2, command=self._open_terms_editor,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side="left", padx=1)

        # ── Toolbar Row 2: Audio source + Device + Status ──────────────
        row2 = tk.Frame(root)
        row2.pack(fill="x", padx=8, pady=(0, 2))

        # Audio source toggle: Loopback / Mic
        self._audio_mode = tk.StringVar(value="loopback")
        tk.Radiobutton(
            row2, text="\U0001f50a Loa", variable=self._audio_mode, value="loopback",
            font=("Segoe UI", 9), fg="#64b5f6", selectcolor="#1a1a2e",
            command=self._refresh_devices,
        ).pack(side="left")
        tk.Radiobutton(
            row2, text="\U0001f399 Mic", variable=self._audio_mode, value="mic",
            font=("Segoe UI", 9), fg="#ffb74d", selectcolor="#1a1a2e",
            command=self._refresh_devices,
        ).pack(side="left", padx=(0, 4))

        self.devices = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or ["(Không tìm thấy device)"]
        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(row2, textvariable=self.dev_var,
                                      values=dev_names, state="readonly", width=42)
        self.dev_combo.pack(side="left", padx=4)
        default_idx = next(
            (i for i, n in enumerate(dev_names)
             if "Headphone" in n or "Realtek" in n or "Speaker" in n), 0
        )
        self.dev_combo.current(default_idx)

        # TTS output device + voice + speed
        ttk.Separator(row2, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)
        tk.Label(row2, text="\U0001f508TTS→", fg="#ffb74d",
                 font=("Segoe UI", 8)).pack(side="left")

        self._tts_devices = list_output_devices()
        tts_dev_names = [n for _, n in self._tts_devices] or ["(Mặc định)"]
        self._tts_dev_var = tk.StringVar()
        self._tts_dev_combo = ttk.Combobox(
            row2, textvariable=self._tts_dev_var,
            values=tts_dev_names, state="readonly", width=24,
        )
        self._tts_dev_combo.pack(side="left", padx=2)
        self._tts_dev_combo.current(0)

        # Voice selector
        _load_all_tts_voices()
        self._tts_voice_var = tk.StringVar()
        self._tts_voice_combo = ttk.Combobox(
            row2, textvariable=self._tts_voice_var,
            state="readonly", width=20,
        )
        self._tts_voice_combo.pack(side="left", padx=2)
        self._refresh_voice_list()

        # Speed selector
        self._tts_speed_var = tk.StringVar(value="2x")
        speed_labels = [s[0] for s in TTS_SPEED_OPTIONS]
        self._tts_speed_combo = ttk.Combobox(
            row2, textvariable=self._tts_speed_var,
            values=speed_labels, state="readonly", width=4,
        )
        self._tts_speed_combo.pack(side="left", padx=2)
        self._tts_speed_combo.current(2)  # 2x

        # Preview button
        tk.Button(row2, text="▶", width=2, command=self._preview_tts,
                  bg="#21262d", fg="#ffb74d", font=("Segoe UI", 8)).pack(side="left", padx=1)

        self.status = tk.Label(row2, text="Chưa chạy", anchor="w",
                               fg="#555", font=("Segoe UI", 9))
        self.status.pack(side="left", padx=8, fill="x", expand=True)

        # ── Dual Panel: Source (trái) + Target (phải) ─────────────────
        self._panels = tk.PanedWindow(root, orient="horizontal",
                                       sashwidth=4, bg="#30363d")
        self._panels.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Panel Source (left)
        self._src_frame = tk.Frame(self._panels, bg="#0d1117")
        self._src_label = tk.Label(self._src_frame, text="", anchor="w",
                                    bg="#0d1117", fg="#58a6ff",
                                    font=("Segoe UI", 9, "bold"))
        self._src_label.pack(fill="x", padx=4)
        self.text_jp = scrolledtext.ScrolledText(
            self._src_frame, wrap="word", font=("Yu Gothic UI", self._font_size),
            bg="#111", fg="#eee", insertbackground="#eee", relief="flat",
        )
        self.text_jp.pack(fill="both", expand=True)
        self._panels.add(self._src_frame, stretch="always")

        # Panel Target (right)
        self._vi_frame = tk.Frame(self._panels, bg="#0d1117")
        self._tgt_label = tk.Label(self._vi_frame, text="", anchor="w",
                                    bg="#0d1117", fg="#3fb950",
                                    font=("Segoe UI", 9, "bold"))
        self._tgt_label.pack(fill="x", padx=4)
        self.text_vi = scrolledtext.ScrolledText(
            self._vi_frame, wrap="word", font=("Segoe UI", self._font_size),
            bg="#111", fg="#aff3c3", insertbackground="#aff3c3", relief="flat",
        )
        self.text_vi.pack(fill="both", expand=True)
        self._panels.add(self._vi_frame, stretch="always")

        self.text = self.text_jp
        self._update_panel_labels()

        # ── Phím tắt ──────────────────────────────────────────────────
        root.bind("<Control-Shift-C>", lambda e: self.copy_jp())
        root.bind("<Control-Shift-V>", lambda e: self.copy_vi())
        root.bind("<Control-Delete>",  lambda e: self.clear())
        root.bind("<Control-d>",       lambda e: self._toggle_dual())
        root.bind("<Control-t>",       lambda e: self._use_translate.set(not self._use_translate.get()))
        root.bind("<Control-g>",       lambda e: self._open_terms_editor())
        root.bind("<Control-Return>",  lambda e: self.ask_copilot())
        root.bind("<Control-Shift-Return>", lambda e: self.ask_claude())

        self.root.after(100, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── Font control ──────────────────────────────────────────────────
    def _font_up(self):
        self._font_size = min(self._font_size + 2, 40)
        self._apply_font()

    def _font_down(self):
        self._font_size = max(self._font_size - 2, 8)
        self._apply_font()

    def _apply_font(self):
        self.text_jp.config(font=("Yu Gothic UI", self._font_size))
        self.text_vi.config(font=("Segoe UI", self._font_size))

    # ── Language helpers ──────────────────────────────────────────────
    def _get_src_lang(self) -> str:
        idx = self._src_combo.current()
        return self._lang_codes[idx] if idx >= 0 else "ja"

    def _get_tgt_lang(self) -> str:
        idx = self._tgt_combo.current()
        return self._lang_codes[idx] if idx >= 0 else "vi"

    def _get_lang_pair(self) -> tuple[str, str]:
        return self._get_src_lang(), self._get_tgt_lang()

    def _update_panel_labels(self):
        src = SUPPORTED_LANGS.get(self._get_src_lang(), {})
        tgt = SUPPORTED_LANGS.get(self._get_tgt_lang(), {})
        self._src_label.config(text=f"{src.get('flag','')} {src.get('name','Source')}")
        self._tgt_label.config(text=f"{tgt.get('flag','')} {tgt.get('name','Target')}")

    def _on_lang_change(self, event=None):
        self._update_panel_labels()
        self._refresh_voice_list()
        src = self._get_src_lang()
        tgt = self._get_tgt_lang()
        self.root.title(f"Interview STT — {SUPPORTED_LANGS[src]['name']} → {SUPPORTED_LANGS[tgt]['name']}")

    # ── Dual / Single toggle ─────────────────────────────────────────
    def _toggle_dual(self):
        self._dual_mode = not self._dual_mode
        if self._dual_mode:
            self._panels.add(self._vi_frame, stretch="always")
            self.dual_btn.config(text="\U0001f4d6 Dual", bg="#21262d")
        else:
            self._panels.forget(self._vi_frame)
            self.dual_btn.config(text="\U0001f4c4 Single", bg="#21262d")

    # ── Translate toggle ──────────────────────────────────────────────
    def _on_translate_toggle(self, *_):
        self._translate_on = self._use_translate.get()
        if self._translate_on:
            self._ensure_translator()

    def _on_tts_toggle(self, *_):
        self._tts_on = self._use_tts.get()
        if self._tts_on:
            self._ensure_tts()
        elif self.tts_thread:
            self.tts_thread.stop()
            self.tts_thread = None

    def _get_tts_device(self) -> str | None:
        name = self._tts_dev_var.get()
        if name and name != "(Mặc định)":
            return name
        return None

    def _get_tts_voice(self) -> str | None:
        display = self._tts_voice_var.get()
        voices = get_voices_for_lang(self._get_tgt_lang())
        for v in voices:
            gender = "♀" if v["Gender"] == "Female" else "♂"
            label = f"{v['ShortName'].split('-')[-1].replace('Neural','')} {gender}"
            if label == display:
                return v["ShortName"]
        return None

    def _get_tts_rate(self) -> str:
        label = self._tts_speed_var.get()
        for lbl, rate in TTS_SPEED_OPTIONS:
            if lbl == label:
                return rate
        return "+100%"

    def _refresh_voice_list(self):
        lang = self._get_tgt_lang()
        voices = get_voices_for_lang(lang)
        labels = []
        for v in voices:
            gender = "♀" if v["Gender"] == "Female" else "♂"
            labels.append(f"{v['ShortName'].split('-')[-1].replace('Neural','')} {gender}")
        self._tts_voice_combo["values"] = labels or ["(Không có)"]
        if labels:
            default = TTS_DEFAULT_VOICE.get(lang, "")
            idx = next((i for i, v in enumerate(voices) if v["ShortName"] == default), 0)
            self._tts_voice_combo.current(idx)

    def _preview_tts(self):
        if not HAS_TTS:
            self._flash_status("⚠️ edge-tts chưa cài!")
            return
        self._flash_status("\U0001f508 Đang phát thử...")
        threading.Thread(target=self._do_preview_tts, daemon=True).start()

    def _do_preview_tts(self):
        sample_texts = {
            "vi": "Xin chào, cảm ơn bạn đã đến phỏng vấn.",
            "ja": "面接にお越しいただきありがとうございます。",
            "en": "Thank you for coming to the interview.",
            "zh": "感谢您来参加面试。",
            "my": "အင်တာဗျူးလာတဲ့အတွက် ကျေးဇူးတင်ပါတယ်။",
        }
        lang = self._get_tgt_lang()
        text = sample_texts.get(lang, "Hello")
        voice = self._get_tts_voice() or TTS_DEFAULT_VOICE.get(lang, "en-US-AriaNeural")
        rate = self._get_tts_rate()
        try:
            import tempfile
            loop = asyncio.new_event_loop()
            comm = edge_tts.Communicate(text, voice, rate=rate)
            buf = io.BytesIO()
            async def gen():
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
            loop.run_until_complete(gen())
            loop.close()
            if buf.tell() == 0:
                return
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(buf.read())
                tmp = f.name
            _play_mp3_on_device(tmp, self._get_tts_device())
            os.remove(tmp)
        except Exception as e:
            print(f"[TTS Preview] Error: {e}")

    def _ensure_tts(self):
        if not HAS_TTS:
            return
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.tts_thread = TTSThread(
                self.tts_queue, self._get_tgt_lang, self._get_tts_device,
                self._get_tts_voice, self._get_tts_rate,
            )
            self.tts_thread.start()

    def _ensure_translator(self):
        if self.translator_thread is None or not self.translator_thread.is_alive():
            whisper_ready = self.transcriber.model_ready if self.transcriber else None
            self.translator_thread = TranslatorThread(
                self.jp_trans_queue, self.vi_queue, self.err_queue,
                status_queue=self.status_queue,
                wait_for_event=whisper_ready,
                get_lang_pair=self._get_lang_pair,
            )
            self.translator_thread.start()

    # ── Smart Scroll ──────────────────────────────────────────────────
    def _is_near_bottom(self, widget):
        try:
            _, yview_end = widget.yview()
            return yview_end > 0.95
        except Exception:
            return True

    def _smart_scroll(self, widget):
        if self._is_near_bottom(widget):
            widget.see("end")

    # ── Device ────────────────────────────────────────────────────────
    def _refresh_devices(self):
        mode = self._audio_mode.get()
        if mode == "mic":
            self.devices = list_mic_devices()
        else:
            self.devices = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or [f"(Không tìm thấy {mode})"]
        self.dev_combo["values"] = dev_names
        if dev_names:
            self.dev_combo.current(0)

    def _selected_device_index(self):
        sel = self.dev_var.get()
        for idx, name in self.devices:
            if name == sel:
                return idx
        return None

    def _load_diarization(self):
        load_diarization_pipeline()
        if HAS_DIARIZATION:
            self.status_queue.put("Diarization OK. Đang nghe loa...")

    # ── Start / Stop ──────────────────────────────────────────────────
    def toggle(self):
        if not self.running:
            self.start()
        else:
            self.stop()

    def start(self):
        self.running = True
        self.btn.config(text="■ Stop", bg="#c62828")
        self.dev_combo.config(state="disabled")
        self._src_combo.config(state="disabled")
        self._tgt_combo.config(state="disabled")

        if not HAS_DIARIZATION and HF_TOKEN:
            self.set_status("Đang tải diarization model...")
            threading.Thread(target=self._load_diarization, daemon=True).start()

        if self._audio_mode.get() == "mic":
            self.audio = AudioMic(self.frame_queue,
                                  device_index=self._selected_device_index())
        else:
            self.audio = AudioLoopback(self.frame_queue,
                                       device_index=self._selected_device_index())
        try:
            self.audio.start()
        except Exception as e:
            self.set_status(f"Lỗi audio: {e}")
            self.running = False
            self.btn.config(text="▶ Start", bg="#2e7d32")
            self.dev_combo.config(state="readonly")
            self._src_combo.config(state="readonly")
            self._tgt_combo.config(state="readonly")
            return

        self.transcriber = Transcriber(
            self.frame_queue, self.text_queue, self.status_queue,
            lambda: (self.audio.device_rate, self.audio.channels),
            use_diarization=self._use_diarization,
            get_src_lang=self._get_src_lang,
        )
        self.transcriber.start()

        # Auto-start translator nếu dịch đang bật
        if self._translate_on:
            self._ensure_translator()

    def stop(self):
        self.running = False
        self.btn.config(text="▶ Start", bg="#2e7d32")
        self.dev_combo.config(state="readonly")
        self._src_combo.config(state="readonly")
        self._tgt_combo.config(state="readonly")
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
        self._last_speaker = None
        _speaker_registry.reset()
        self.set_status("Đã dừng")

    # ── Clear / Copy ──────────────────────────────────────────────────
    def clear(self):
        self.text_jp.delete("1.0", "end")
        self.text_vi.delete("1.0", "end")
        self._pending_vi.clear()
        self._seg_counter = 0
        self._vi_next_id = 0

    def copy_jp(self):
        content = self.text_jp.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._flash_status("✅ Đã copy JP!")

    def copy_vi(self):
        content = self.text_vi.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._flash_status("✅ Đã copy VI!")

    # ── AI automation (generic) ──────────────────────────────────────
    def _select_ai(self, name, cfg):
        self._ai_choice = name
        self._ai_send_btn.config(text=f"\U0001f916 {name}", bg=cfg["color"])
        self._ai_drop.config(bg=cfg["color"], activebackground=cfg["color"])

    def _send_to_ai(self):
        reg = _AI_REGISTRY.get(self._ai_choice)
        if reg:
            open_fn, find_fn, click_fn, delay = reg
            self._ask_ai(self._ai_choice, open_fn, find_fn, click_fn, delay)

    def ask_copilot(self):
        self._ask_ai("Copilot", open_or_focus_copilot,
                      find_copilot_hwnd, click_copilot_input)

    def ask_claude(self):
        self._ask_ai("Claude", open_or_focus_claude,
                      find_claude_hwnd, click_claude_input, launch_delay=5000)

    def ask_chatgpt(self):
        self._ask_ai("ChatGPT", open_or_focus_chatgpt,
                      find_chatgpt_hwnd, click_chatgpt_input)

    def _ask_ai(self, name, open_fn, find_fn, click_fn, launch_delay=4500):
        content = self.text_jp.get("1.0", "end").strip()
        if not content:
            self._flash_status("⚠️ Chưa có text nào để hỏi!")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        hwnd = open_fn()
        self._ai_target = (name, find_fn, click_fn)
        if hwnd:
            self._flash_status(f"⏳ Đang gửi sang {name}...")
            self.root.after(400, self._ai_click_and_paste)
        else:
            self._flash_status(f"\U0001f680 Đang mở {name}...")
            self.root.after(launch_delay, self._ai_click_and_paste)

    def _ai_click_and_paste(self):
        name, find_fn, click_fn = self._ai_target
        hwnd = find_fn()
        if hwnd:
            click_fn(hwnd)
        self.root.after(350, self._ai_do_paste)

    def _ai_do_paste(self):
        user32 = ctypes.windll.user32
        KEYUP = 0x0002
        user32.keybd_event(0x11, 0, 0, 0)
        user32.keybd_event(0x56, 0, 0, 0)
        user32.keybd_event(0x56, 0, KEYUP, 0)
        user32.keybd_event(0x11, 0, KEYUP, 0)
        self.root.after(300, self._ai_do_enter)

    def _ai_do_enter(self):
        user32 = ctypes.windll.user32
        KEYUP = 0x0002
        user32.keybd_event(0x0D, 0, 0, 0)
        user32.keybd_event(0x0D, 0, KEYUP, 0)
        name = self._ai_target[0]
        self._flash_status(f"✅ Đã gửi! {name} đang trả lời...")

    # ── Status helpers ────────────────────────────────────────────────
    def _flash_status(self, msg):
        old = self.status.cget("text")
        self.status.config(text=msg, fg="#00e676")
        self.root.after(2000, lambda: self.status.config(text=old, fg="#555"))

    def set_status(self, msg):
        self.status.config(text=msg, fg="#555")

    # ── Poll queues ───────────────────────────────────────────────────
    def poll(self):
        while not self.status_queue.empty():
            self.set_status(self.status_queue.get())

        # JP text từ Whisper
        while not self.text_queue.empty():
            item    = self.text_queue.get()
            jp_text = item["text"].strip()
            color   = item.get("color", "#eee")
            speaker = item.get("speaker")

            tag = f"spk_{color.replace('#','')}"
            self.text_jp.tag_configure(tag, foreground=color)

            same_speaker = (speaker == self._last_speaker)
            if not same_speaker:
                if self._last_speaker is not None:
                    self.text_jp.insert("end", "\n", tag)
                if speaker:
                    self.text_jp.insert("end", f"{speaker}\n", tag)
                self.text_jp.insert("end", jp_text, tag)
                self._last_speaker = speaker
            else:
                last_char = self.text_jp.get("end-2c", "end-1c")
                if last_char in ("。", "！", "？", ".", "!", "?"):
                    self.text_jp.insert("end", "\n" + jp_text, tag)
                else:
                    self.text_jp.insert("end", jp_text, tag)

            self._smart_scroll(self.text_jp)

            # FIFO: gửi (seg_id, jp_text) qua translator
            if self._translate_on:
                seg_id = self._seg_counter
                self._seg_counter += 1
                self.jp_trans_queue.put((seg_id, jp_text))
                # Hiện placeholder trong panel VI
                self.text_vi.insert("end", "⏳ ...\n", "pending")
                self.text_vi.tag_configure("pending", foreground="#555")
                self._pending_vi[seg_id] = self.text_vi.index("end-2l linestart")
                self._smart_scroll(self.text_vi)

        # VI text từ Qwen / Google — FIFO thay thế placeholder
        while not self.vi_queue.empty():
            result = self.vi_queue.get()
            if isinstance(result, tuple):
                seg_id, vi_text = result
            else:
                seg_id, vi_text = None, result

            if seg_id is not None and seg_id in self._pending_vi:
                pos = self._pending_vi.pop(seg_id)
                try:
                    line_end = f"{pos} lineend"
                    self.text_vi.delete(pos, line_end)
                    self.text_vi.insert(pos, vi_text, "translated")
                    self.text_vi.tag_configure("translated", foreground="#aff3c3")
                except Exception:
                    self.text_vi.insert("end", vi_text + "\n", "translated")
            else:
                self.text_vi.insert("end", vi_text + "\n", "translated")
                self.text_vi.tag_configure("translated", foreground="#aff3c3")

            if self._tts_on and vi_text.strip():
                self.tts_queue.put(vi_text)

            self._smart_scroll(self.text_vi)

        # Lỗi dịch
        while not self.err_queue.empty():
            err = self.err_queue.get()
            self.set_status(f"⚠️ {err}")

        self.root.after(100, self.poll)

    # ── Terms Editor ────────────────────────────────────────────────
    def _open_terms_editor(self):
        import translation.terms as terms_mod

        win = tk.Toplevel(self.root)
        win.title("⚙ Thuật ngữ phỏng vấn (JP → VI)")
        win.geometry("500x420")
        win.attributes("-topmost", True)

        tk.Label(win, text="Mỗi dòng: 日本語 = Tiếng Việt",
                 font=("Segoe UI", 10), fg="#888").pack(padx=10, pady=(8, 2), anchor="w")

        txt = scrolledtext.ScrolledText(
            win, wrap="word", font=("Segoe UI", 12),
            bg="#111", fg="#eee", insertbackground="#eee",
        )
        txt.pack(fill="both", expand=True, padx=10, pady=4)

        # Load terms hiện tại
        for jp, vi in terms_mod._custom_terms.items():
            txt.insert("end", f"{jp} = {vi}\n")

        status_lbl = tk.Label(win, text="", fg="#888", font=("Segoe UI", 9))
        status_lbl.pack(padx=10, anchor="w")

        def save():
            content = txt.get("1.0", "end").strip()
            new_terms = {}
            for line in content.splitlines():
                line = line.strip()
                if "=" in line:
                    parts = line.split("=", 1)
                    jp = parts[0].strip()
                    vi = parts[1].strip()
                    if jp and vi:
                        new_terms[jp] = vi
            terms_mod._custom_terms.clear()
            terms_mod._custom_terms.update(new_terms)
            try:
                with open(TERMS_PATH, "w", encoding="utf-8") as f:
                    json.dump(terms_mod._custom_terms, f, ensure_ascii=False, indent=2)
            except Exception as e:
                status_lbl.config(text=f"❌ Lỗi lưu: {e}", fg="#ef5350")
                return
            # Cập nhật hint trong Qwen translator nếu đang chạy
            if (self.translator_thread and hasattr(self.translator_thread, '_qwen')
                    and self.translator_thread._qwen):
                self.translator_thread._qwen._terms_hint = _build_terms_hint()
            status_lbl.config(text=f"✅ Đã lưu {len(new_terms)} thuật ngữ!", fg="#3fb950")
            win.after(2000, lambda: status_lbl.config(text="", fg="#888"))

        btn_bar = tk.Frame(win)
        btn_bar.pack(fill="x", padx=10, pady=(0, 8))
        tk.Button(btn_bar, text="\U0001f4be Lưu", width=10, command=save,
                  bg="#238636", fg="white", font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Button(btn_bar, text="Đóng", width=8, command=win.destroy,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 10)).pack(side="left", padx=6)
        tk.Label(btn_bar, text=f"\U0001f4c1 {TERMS_PATH}",
                 fg="#555", font=("Segoe UI", 8)).pack(side="right")

    def on_close(self):
        self.stop()
        if self.translator_thread:
            self.translator_thread.stop()
        if self.tts_thread:
            self.tts_thread.stop()
        self.root.destroy()
