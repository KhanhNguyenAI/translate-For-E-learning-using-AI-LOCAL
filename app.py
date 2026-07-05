# -*- coding: utf-8 -*-
"""
Class App — main PySide6 UI, poll loops, orchestration.
"""

import os
import json
import ctypes
import threading

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QRadioButton, QButtonGroup, QTextEdit, QSplitter,
    QFrame, QToolButton, QMenu, QDialog, QApplication, QDockWidget, QFormLayout,
    QFileDialog, QSizePolicy, QProgressBar, QScrollArea,
)
from PySide6.QtGui import (
    QColor, QFont, QTextCharFormat, QTextCursor, QShortcut, QKeySequence,
)
from PySide6.QtCore import Qt, QTimer

from config import (
    SUPPORTED_LANGS, DEFAULT_FONT_SIZE, HF_TOKEN, RECORD_DIR,
    INLINE_FROM, INLINE_TO, INLINE_ENGINE, INLINE_ENABLED,
)
from inline_translate import GlobalHotkey, InlineTranslator
from system_monitor import SystemMonitor
from mindmap import MindmapWorker, MindmapDialog
from audio import AudioLoopback, AudioMic, list_loopback_devices, list_mic_devices
from stt import (
    Transcriber, ReazonSpeechTranscriber, load_diarization_pipeline, _speaker_registry,
)
import stt.diarization as _diar_mod
from translation import TERMS_PATH, _custom_terms, _build_terms_hint
from translation.qwen import TranslatorThread
from chat_dialog import ChatDialog
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

import queue


# ── Dark theme (QSS) ──────────────────────────────────────────────────
DARK_QSS = """
QMainWindow, QWidget {
    background: #0d1117; color: #c9d1d9;
    font-family: 'Segoe UI'; font-size: 12px;
}

/* ── Combo boxes ── */
QComboBox {
    background:#1c2128; border:1px solid #30363d; border-radius:6px;
    padding:4px 8px; color:#c9d1d9; min-height:18px;
}
QComboBox:hover { border:1px solid #484f58; }
QComboBox:focus { border:1px solid #58a6ff; }
QComboBox:disabled { color:#484f58; background:#161b22; }
QComboBox::drop-down { border:none; width:18px; }
QComboBox::down-arrow {
    image:none; width:0; height:0;
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:5px solid #8b949e; margin-right:6px;
}
QComboBox QAbstractItemView {
    background:#161b22; color:#c9d1d9; border:1px solid #30363d;
    border-radius:6px; selection-background-color:#1f6feb; outline:0;
    padding:2px;
}

/* ── Buttons ── */
QPushButton {
    background:#21262d; border:1px solid #30363d; border-radius:6px;
    padding:5px 10px; color:#c9d1d9;
}
QPushButton:hover { background:#30363d; border:1px solid #484f58; }
QPushButton:pressed { background:#282e36; }
QPushButton:disabled { color:#484f58; background:#161b22; }

/* ── AI tool button (send + dropdown) ── */
QToolButton {
    background:#0078d4; border:none; border-radius:6px;
    padding:5px 10px; color:white; font-weight:bold;
}
QToolButton:hover { background:#1184e0; }
QToolButton::menu-button {
    width:18px; border:none; background:rgba(0,0,0,0.18);
    border-top-right-radius:6px; border-bottom-right-radius:6px;
}
QToolButton::menu-button:hover { background:rgba(0,0,0,0.32); }
QToolButton::menu-arrow {
    image:none; width:0; height:0;
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:5px solid white;
}

/* ── Menus ── */
QMenu { background:#1c2128; color:#c9d1d9; border:1px solid #30363d; border-radius:8px; padding:4px; }
QMenu::item { padding:6px 18px; border-radius:5px; }
QMenu::item:selected { background:#1f6feb; color:white; }

/* ── Checkboxes & radios ── */
QCheckBox, QRadioButton { color:#aaa; spacing:4px; }
QCheckBox::indicator, QRadioButton::indicator { width:15px; height:15px; }
QCheckBox::indicator {
    border:1px solid #484f58; border-radius:4px; background:#1c2128;
}
QCheckBox::indicator:checked { background:#1f6feb; border:1px solid #1f6feb; }
QRadioButton::indicator {
    border:1px solid #484f58; border-radius:8px; background:#1c2128;
}
QRadioButton::indicator:checked { background:#1f6feb; border:3px solid #1c2128; }

/* ── Text panels ── */
QTextEdit {
    background:#0a0c10; color:#eee; border:1px solid #1c2128;
    border-radius:8px; padding:6px; selection-background-color:#264f78;
}

/* ── Scrollbars (slim) ── */
QScrollBar:vertical { background:transparent; width:9px; margin:2px; }
QScrollBar::handle:vertical { background:#30363d; border-radius:4px; min-height:24px; }
QScrollBar::handle:vertical:hover { background:#484f58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
QScrollBar:horizontal { background:transparent; height:9px; margin:2px; }
QScrollBar::handle:horizontal { background:#30363d; border-radius:4px; min-width:24px; }
QScrollBar::handle:horizontal:hover { background:#484f58; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background:none; }

QLabel { color:#c9d1d9; }

/* ── Splitter ── */
QSplitter::handle { background:#0d1117; }
QSplitter::handle:horizontal { width:6px; }
QSplitter::handle:hover { background:#1f6feb; }

QDialog { background:#0d1117; }

/* ── Redesign: language pill, toggle group, chat button ── */
QFrame#langPill { background:#1c2128; border:1px solid #2a3038; border-radius:16px; }
QFrame#tgroup { background:#1c2128; border:1px solid #2a3038; border-radius:8px; }
QPushButton#swap { background:transparent; border:none; color:#6e7681; padding:0 4px; font-size:14px; }
QPushButton#swap:hover { color:#c9d1d9; }
QPushButton#chatBtn { background:rgba(210,168,255,0.12); border:1px solid rgba(210,168,255,0.30); border-radius:7px; color:#d2a8ff; padding:5px 11px; }
QPushButton#chatBtn:hover { background:rgba(210,168,255,0.22); }
QToolButton#aiMenu { background:rgba(210,168,255,0.12); border:1px solid rgba(210,168,255,0.30); border-radius:7px; color:#d2a8ff; padding:5px 12px; font-weight:500; }
QToolButton#aiMenu:hover { background:rgba(210,168,255,0.20); }
QToolButton#aiMenu::menu-indicator { image:none; width:0; }
QPushButton#gear { background:#1c2128; border:1px solid #2a3038; border-radius:7px; color:#8b949e; }
QPushButton#gear:hover { background:#30363d; color:#c9d1d9; }

/* ── Settings drawer ── */
QDockWidget { color:#c9d1d9; titlebar-close-icon:none; titlebar-normal-icon:none; }
QDockWidget::title { background:#161b22; padding:7px 10px; border-bottom:1px solid #2a3038; }
QScrollArea { border:none; background:transparent; }
QWidget#section { border:1px solid #2a3038; border-radius:9px; }
QPushButton#secHead { text-align:left; background:#1c2128; border:none;
    border-top-left-radius:8px; border-top-right-radius:8px; padding:8px 11px;
    color:#c9d1d9; font-weight:500; font-size:12px; }
QPushButton#secHead:hover { background:#22272e; }
QStatusBar { background:#161b22; border-top:1px solid #2a3038; }
QStatusBar::item { border:none; }

/* ── System monitor ── */
QProgressBar { background:#161b22; border:1px solid #30363d; border-radius:5px; height:16px; text-align:center; color:#c9d1d9; font-size:10px; }
QProgressBar::chunk { border-radius:4px; }
QPushButton#sysmon { background:transparent; border:none; color:#8b949e; font-size:11px; padding:0 8px; }
QPushButton#sysmon:hover { color:#c9d1d9; }
"""

# Flat combos used inside the language pill (no border / no arrow box)
FLAT_COMBO = (
    "QComboBox{background:transparent;border:none;padding:2px 2px;color:#c9d1d9;}"
    "QComboBox::drop-down{width:0;border:none;}"
    "QComboBox QAbstractItemView{background:#161b22;color:#c9d1d9;"
    "border:1px solid #30363d;selection-background-color:#1f6feb;}"
)


class _BoolVarAdapter:
    """Provides a .get() interface (Tkinter BooleanVar compat) over a Qt getter."""
    def __init__(self, getter):
        self._getter = getter

    def get(self):
        return self._getter()


class _Section(QWidget):
    """Collapsible settings section: a clickable header + a QFormLayout body."""
    def __init__(self, title, icon="", expanded=True, parent=None):
        super().__init__(parent)
        self.setObjectName("section")
        self._icon = icon
        self._title = title
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._btn = QPushButton()
        self._btn.setObjectName("secHead")
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.toggled.connect(self._sync)
        v.addWidget(self._btn)
        self._body = QWidget()
        self.form = QFormLayout(self._body)
        self.form.setContentsMargins(11, 8, 11, 10)
        self.form.setSpacing(8)
        v.addWidget(self._body)
        self._sync(expanded)

    def _sync(self, on):
        self._body.setVisible(on)
        self._btn.setText(f"{'▾' if on else '▸'}   {self._icon}  {self._title}")

    def addRow(self, *a):
        self.form.addRow(*a)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interview STT — Multi-Language")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setGeometry(40, 40, 1000, 600)
        self.setMinimumSize(880, 440)
        self.setStyleSheet(DARK_QSS)

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
        self._jp_last_char = ""
        self._font_size = DEFAULT_FONT_SIZE
        self._dual_mode = True
        self._translate_on = False
        self._tts_on = False
        self._seg_counter = 0
        self._pending_vi = {}
        self._vi_next_id = 0
        self._ai_choice = "Copilot"
        self._recorder = None
        self._record_dir = RECORD_DIR

        # ── Language dropdown options ─────────────────────────────────
        self._lang_codes = list(SUPPORTED_LANGS.keys())
        lang_display = [f"{SUPPORTED_LANGS[c]['flag']} {SUPPORTED_LANGS[c]['name']}"
                        for c in self._lang_codes]

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 6, 8, 8)
        root_layout.setSpacing(4)

        # ── Single-row toolbar ────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        root_layout.addLayout(row1)

        # - Language pill: src ⇄ tgt -
        pill = QFrame(); pill.setObjectName("langPill")
        pill_lay = QHBoxLayout(pill)
        pill_lay.setContentsMargins(8, 1, 8, 1); pill_lay.setSpacing(2)
        self._src_combo = QComboBox()
        self._src_combo.addItems(lang_display)
        self._src_combo.setCurrentIndex(0)  # ja
        self._src_combo.setStyleSheet(FLAT_COMBO)
        self._src_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._src_combo.setMinimumWidth(72)
        self._tgt_combo = QComboBox()
        self._tgt_combo.addItems(lang_display)
        self._tgt_combo.setCurrentIndex(self._lang_codes.index("vi"))
        self._tgt_combo.setStyleSheet(FLAT_COMBO)
        self._tgt_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._tgt_combo.setMinimumWidth(72)
        swap = QPushButton("⇄"); swap.setObjectName("swap")
        swap.setToolTip("Swap languages"); swap.clicked.connect(self._swap_langs)
        pill_lay.addWidget(self._src_combo)
        pill_lay.addWidget(swap)
        pill_lay.addWidget(self._tgt_combo)
        row1.addWidget(pill)

        row1.addWidget(self._bar_label("Input"))

        # - Toggle group: Translate / TTS / Speakers -
        tgroup = QFrame(); tgroup.setObjectName("tgroup")
        tg_lay = QHBoxLayout(tgroup)
        tg_lay.setContentsMargins(3, 3, 3, 3); tg_lay.setSpacing(2)
        self._cb_translate = self._make_toggle("\U0001f310 Translate", "#58a6ff")
        self._cb_translate.toggled.connect(self._on_translate_toggle)
        self._cb_tts = self._make_toggle("\U0001f508 TTS", "#d29922")
        self._cb_tts.setEnabled(HAS_TTS)
        self._cb_tts.toggled.connect(self._on_tts_toggle)
        self._cb_diar = self._make_toggle("\U0001f465 Speakers", "#3fb950")
        tg_lay.addWidget(self._cb_translate)
        tg_lay.addWidget(self._cb_tts)
        tg_lay.addWidget(self._cb_diar)
        row1.addWidget(tgroup)
        self._use_diarization = _BoolVarAdapter(self._cb_diar.isChecked)

        row1.addStretch()

        # - AI tools: menu (Chat · Map) -
        row1.addWidget(self._bar_label("AI tools"))
        self._ai_menu_btn = QToolButton()
        self._ai_menu_btn.setObjectName("aiMenu")
        self._ai_menu_btn.setText("\U0001f9e0 AI  ▾")
        self._ai_menu_btn.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self._ai_menu_btn)
        m.addAction("\U0001f9e0   AI Chat").triggered.connect(lambda checked=False: self._open_chat())
        m.addAction("\U0001f5fa   Mind map").triggered.connect(lambda checked=False: self._make_mindmap())
        self._ai_menu_btn.setMenu(m)
        row1.addWidget(self._ai_menu_btn)

        # - Send to external AI: split button (click = send, ▾ = pick model) -
        self._ai_btn = QToolButton()
        self._ai_btn.setText("\U0001f916 Copilot")
        self._ai_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._ai_btn.clicked.connect(self._send_to_ai)
        ai_menu = QMenu(self._ai_btn)
        for ai_name, ai_cfg in AI_OPTIONS.items():
            act = ai_menu.addAction(f"{ai_cfg['icon']}  {ai_name}")
            act.triggered.connect(
                lambda checked=False, n=ai_name, c=ai_cfg: self._select_ai(n, c))
        self._ai_btn.setMenu(ai_menu)
        row1.addWidget(self._ai_btn)

        # - Start -
        self.btn = QPushButton("▶ Start")
        self.btn.setStyleSheet("background:#238636; color:white; font-weight:500; border:none;")
        self.btn.clicked.connect(self.toggle)
        row1.addWidget(self.btn)

        # - Record transcript -
        self._rec_btn = QPushButton("● Record")
        self._rec_btn.setToolTip("Save the source transcript to a file (not audio)")
        self._rec_btn.clicked.connect(self._toggle_record)
        row1.addWidget(self._rec_btn)

        # - Settings gear -
        self._gear_btn = QPushButton("⚙"); self._gear_btn.setObjectName("gear")
        self._gear_btn.setFixedWidth(34); self._gear_btn.setToolTip("Settings")
        self._gear_btn.clicked.connect(self._toggle_settings)
        row1.addWidget(self._gear_btn)

        # ── Settings drawer (all the set-once controls) ───────────────
        self._build_settings_dock()

        # ── Status bar ────────────────────────────────────────────────
        self.status = QLabel("Idle")
        self.status.setStyleSheet("color:#8b949e; font-size:11px;")
        self.statusBar().addWidget(self.status, 1)

        self._sysmon_btn = QPushButton("…")
        self._sysmon_btn.setObjectName("sysmon")
        self._sysmon_btn.setToolTip("System monitor — click for details")
        self._sysmon_btn.clicked.connect(self._toggle_settings)
        self.statusBar().addPermanentWidget(self._sysmon_btn)

        self._refresh_voice_list()

        # ── Panels: Source + Target + AI Analysis ─────────────────────
        self._panels = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self._panels, 1)

        # Panel Source (left)
        self._src_panel = QWidget()
        src_lay = QVBoxLayout(self._src_panel)
        src_lay.setContentsMargins(0, 0, 0, 0); src_lay.setSpacing(2)
        src_head = QHBoxLayout(); src_head.setContentsMargins(4, 2, 4, 0); src_head.setSpacing(2)
        self._src_label = QLabel("")
        self._src_label.setStyleSheet("color:#58a6ff; font-weight:bold; padding:2px 4px;")
        src_head.addWidget(self._src_label)
        src_head.addStretch()
        src_head.addWidget(self._icon_btn("\U0001f4cb", "Copy source", self.copy_jp))
        src_head.addWidget(self._icon_btn("\U0001f5d1", "Clear all", self.clear))
        src_lay.addLayout(src_head)
        self.text_jp = QTextEdit()
        self.text_jp.setReadOnly(False)   # editable — paste text to test Map / AI Chat
        self.text_jp.setPlaceholderText(
            "Transcript appears here…\nYou can also type or paste text directly "
            "to try 🗺 Map, 🧠 AI Chat, or the AI send button.")
        self.text_jp.setStyleSheet("background:#111; color:#eee;")
        self.text_jp.setFont(QFont("Yu Gothic UI", self._font_size))
        src_lay.addWidget(self.text_jp)
        self._panels.addWidget(self._src_panel)

        # Panel Target (middle)
        self._vi_panel = QWidget()
        vi_lay = QVBoxLayout(self._vi_panel)
        vi_lay.setContentsMargins(0, 0, 0, 0); vi_lay.setSpacing(2)
        vi_head = QHBoxLayout(); vi_head.setContentsMargins(4, 2, 4, 0); vi_head.setSpacing(2)
        self._tgt_label = QLabel("")
        self._tgt_label.setStyleSheet("color:#3fb950; font-weight:bold; padding:2px 4px;")
        vi_head.addWidget(self._tgt_label)
        vi_head.addStretch()
        vi_head.addWidget(self._icon_btn("\U0001f4cb", "Copy translation", self.copy_vi))
        vi_lay.addLayout(vi_head)
        self.text_vi = QTextEdit()
        self.text_vi.setReadOnly(True)
        self.text_vi.setStyleSheet("background:#111; color:#aff3c3;")
        self.text_vi.setFont(QFont("Segoe UI", self._font_size))
        vi_lay.addWidget(self.text_vi)
        self._panels.addWidget(self._vi_panel)
        self._panels.setChildrenCollapsible(False)
        self._panels.setStretchFactor(0, 1)
        self._panels.setStretchFactor(1, 1)
        self._panels.setSizes([500, 500])

        self._qwen_instance = None
        self._chat_dialog = None

        self.text = self.text_jp
        self._update_panel_labels()

        # ── Connect language change signals (after build) ─────────────
        self._src_combo.currentIndexChanged.connect(self._on_lang_change)
        self._tgt_combo.currentIndexChanged.connect(self._on_lang_change)

        # ── Keyboard shortcuts ────────────────────────────────────────
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self.copy_jp)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self, self.copy_vi)
        QShortcut(QKeySequence("Ctrl+Delete"),  self, self.clear)
        QShortcut(QKeySequence("Ctrl+D"),       self, self._toggle_dual)
        QShortcut(QKeySequence("Ctrl+T"),       self,
                  lambda: self._cb_translate.setChecked(not self._cb_translate.isChecked()))
        QShortcut(QKeySequence("Ctrl+G"),       self, self._open_terms_editor)
        QShortcut(QKeySequence("Ctrl+Return"),  self, self.ask_copilot)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self, self.ask_claude)

        # ── Inline translate (global hotkeys) ─────────────────────────
        #   Ctrl+Alt+T = replace selection in place
        #   Ctrl+Alt+D = show translation in a floating popup (non-destructive)
        self._inline = InlineTranslator(self._inline_translate_text, self)
        self._inline.status.connect(self._flash_status)
        self._hotkey = GlobalHotkey(0x0002 | 0x0001, 0x54, self)   # Ctrl+Alt+T
        self._hotkey.activated.connect(lambda: self._inline.trigger("replace"))
        self._hotkey.failed.connect(lambda m: self._flash_status(f"⚠️ {m}"))
        self._hotkey2 = GlobalHotkey(0x0002 | 0x0001, 0x44, self)  # Ctrl+Alt+D
        self._hotkey2.activated.connect(lambda: self._inline.trigger("popup"))
        self._hotkey2.failed.connect(lambda m: self._flash_status(f"⚠️ {m}"))
        if self._cb_inline.isChecked():
            self._hotkey.start()
            self._hotkey2.start()
        self._maybe_warm_gemini()

        # ── System monitor ────────────────────────────────────────────
        self._sysmon = SystemMonitor(2.0, self)
        self._sysmon.updated.connect(self._on_sysmon)
        if self._cb_sysmon.isChecked():
            self._sysmon.start()

        # ── Poll timer ────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(100)

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _vsep():
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setStyleSheet("color:#30363d;")
        return f

    @staticmethod
    def _bar_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#6e7681; font-size:10px; padding:0 3px;")
        return lbl

    @staticmethod
    def _make_toggle(text, accent):
        b = QPushButton(text)
        b.setCheckable(True)
        b.setStyleSheet(
            "QPushButton{background:transparent;border:none;border-radius:6px;"
            "padding:5px 10px;color:#8b949e;}"
            f"QPushButton:checked{{background:#0d1117;color:{accent};}}"
            "QPushButton:disabled{color:#484f58;}"
        )
        return b

    def _icon_btn(self, glyph, tip, slot):
        b = QPushButton(glyph)
        b.setFixedSize(26, 22)
        b.setToolTip(tip)
        b.setStyleSheet(
            "QPushButton{background:transparent;border:none;color:#6e7681;}"
            "QPushButton:hover{color:#c9d1d9;}"
        )
        b.clicked.connect(slot)
        return b

    def _swap_langs(self):
        s = self._src_combo.currentIndex()
        t = self._tgt_combo.currentIndex()
        self._src_combo.setCurrentIndex(t)
        self._tgt_combo.setCurrentIndex(s)

    def _toggle_settings(self):
        self._settings_dock.setVisible(not self._settings_dock.isVisible())

    # ── System monitor ───────────────────────────────────────────────
    @staticmethod
    def _usage_color(pct):
        return "#f85149" if pct >= 90 else ("#d29922" if pct >= 70 else "#3fb950")

    def _set_bar(self, pb, pct, text):
        pct = max(0, min(100, int(round(pct))))
        pb.setValue(pct)
        pb.setFormat(text)
        pb.setStyleSheet(
            f"QProgressBar::chunk{{background:{self._usage_color(pct)};border-radius:4px;}}")

    def _on_sysmon(self, s):
        if "app_ram_gb" in s:
            tot = s.get("sys_ram_total_gb", 0) or 1
            self._set_bar(self._pb_app, s["app_ram_gb"] / tot * 100, f"{s['app_ram_gb']:.1f} GB")
        if "sys_ram_pct" in s:
            self._set_bar(self._pb_sram, s["sys_ram_pct"],
                          f"{s['sys_ram_used_gb']:.1f} / {s['sys_ram_total_gb']:.1f} GB")
        if "vram_pct" in s:
            self._set_bar(self._pb_vram, s["vram_pct"],
                          f"{s['vram_used_gb']:.1f} / {s['vram_total_gb']:.1f} GB")
        else:
            self._pb_vram.setValue(0); self._pb_vram.setFormat("N/A")
        if "gpu_util_pct" in s:
            self._set_bar(self._pb_gpu, s["gpu_util_pct"], f"{s['gpu_util_pct']:.0f}%")
        else:
            self._pb_gpu.setValue(0); self._pb_gpu.setFormat("N/A")
        if "cpu_total_pct" in s:
            self._set_bar(self._pb_cpu, s["cpu_total_pct"], f"{s['cpu_total_pct']:.0f}%")

        # Compact status-bar cluster
        parts = []
        if "app_ram_gb" in s:
            parts.append(f"RAM {s['app_ram_gb']:.1f}GB")
        if "vram_used_gb" in s:
            parts.append(f"VRAM {s['vram_used_gb']:.1f}/{s['vram_total_gb']:.0f}GB")
        if "cpu_total_pct" in s:
            parts.append(f"CPU {s['cpu_total_pct']:.0f}%")
        self._sysmon_btn.setText("  ·  ".join(parts) if parts else "…")
        worst = max([s.get("sys_ram_pct", 0), s.get("vram_pct", 0), s.get("cpu_total_pct", 0)])
        self._sysmon_btn.setStyleSheet(
            f"QPushButton#sysmon{{background:transparent;border:none;font-size:11px;"
            f"padding:0 8px;color:{self._usage_color(worst)};}}")

    def _on_sysmon_toggle(self, on):
        if on:
            self._sysmon.start()
            self._sysmon_btn.setVisible(True)
        else:
            self._sysmon.stop()
            self._sysmon_btn.setVisible(False)

    def _choose_record_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose record folder", self._record_dir or "")
        if d:
            self._record_dir = d
            self._rec_dir_btn.setText(d)
            self._rec_dir_btn.setToolTip(d)
            from config import save_config_value
            save_config_value("record_dir", d)

    def _toggle_record(self):
        if self._recorder is None:
            if not self._record_dir:
                self._choose_record_dir()
                if not self._record_dir:
                    return
            if not os.path.isdir(self._record_dir):
                self._flash_status("⚠️ Record folder not found")
                return
            from recorder import RecordWriter
            fmt = self._rec_fmt_combo.currentText()
            src_name = SUPPORTED_LANGS.get(self._get_src_lang(), {}).get("name", "")
            try:
                self._recorder = RecordWriter(self._record_dir, fmt, f"Source: {src_name}")
            except Exception as e:
                self._flash_status(f"⚠️ Record error: {e}")
                return
            self._rec_btn.setText("● Recording")
            self._rec_btn.setStyleSheet(
                "background:rgba(248,81,73,0.18); color:#f85149; border:1px solid #f85149;")
            self._flash_status(f"⏺ Recording → {os.path.basename(self._recorder.path)}")
        else:
            path = self._recorder.path
            self._recorder.close()
            self._recorder = None
            self._rec_btn.setText("● Record")
            self._rec_btn.setStyleSheet("")
            self._flash_status(f"💾 Saved: {os.path.basename(path)}")

    def _build_settings_dock(self):
        dock = QDockWidget("  Settings", self)
        dock.setObjectName("settingsDock")
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # ═══ Section: Speech-to-Text ═══════════════════════════════════
        sec_stt = _Section("Speech-to-Text", "\U0001f3a4", expanded=True)
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(["Whisper", "ReazonSpeech"])
        sec_stt.addRow("Engine", self._engine_combo)

        self._chunk_combo = QComboBox()
        self._chunk_combo.addItems(["Auto", "1s", "2s", "4s", "6s", "8s", "10s"])
        self._chunk_combo.setCurrentText("4s")
        self._chunk_combo.setToolTip("Auto = wait for the speaker to pause, then translate a full sentence")
        sec_stt.addRow("Chunk size", self._chunk_combo)

        src_row = QWidget(); sl = QHBoxLayout(src_row)
        sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(8)
        self._rb_speaker = QRadioButton("\U0001f50a Speaker")
        self._rb_speaker.setChecked(True)
        self._rb_mic = QRadioButton("\U0001f399 Mic")
        audio_group = QButtonGroup(self)
        audio_group.addButton(self._rb_speaker)
        audio_group.addButton(self._rb_mic)
        self._rb_speaker.toggled.connect(self._refresh_devices)
        sl.addWidget(self._rb_speaker); sl.addWidget(self._rb_mic); sl.addStretch()
        sec_stt.addRow("Audio source", src_row)

        self.devices = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or ["(No device found)"]
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(dev_names)
        self.dev_combo.setMinimumWidth(90)
        default_idx = next(
            (i for i, n in enumerate(dev_names)
             if "Headphone" in n or "Realtek" in n or "Speaker" in n), 0
        )
        self.dev_combo.setCurrentIndex(default_idx)
        sec_stt.addRow("Input device", self.dev_combo)
        outer.addWidget(sec_stt)

        # ═══ Section: Text-to-Speech ═══════════════════════════════════
        sec_tts = _Section("Text-to-Speech", "\U0001f508", expanded=False)
        self._tts_devices = list_output_devices()
        tts_dev_names = [n for _, n in self._tts_devices] or ["(Default)"]
        self._tts_dev_combo = QComboBox()
        self._tts_dev_combo.addItems(tts_dev_names)
        self._tts_dev_combo.setMinimumWidth(90)
        self._tts_dev_combo.setCurrentIndex(0)
        sec_tts.addRow("Device", self._tts_dev_combo)

        _load_all_tts_voices()
        self._tts_voice_combo = QComboBox()
        self._tts_voice_combo.setMinimumWidth(90)
        sec_tts.addRow("Voice", self._tts_voice_combo)

        speed_row = QWidget(); spl = QHBoxLayout(speed_row)
        spl.setContentsMargins(0, 0, 0, 0); spl.setSpacing(6)
        self._tts_speed_combo = QComboBox()
        speed_labels = [s[0] for s in TTS_SPEED_OPTIONS]
        self._tts_speed_combo.addItems(speed_labels)
        self._tts_speed_combo.setCurrentIndex(2)  # 2x
        self._tts_speed_combo.setFixedWidth(70)
        b_preview = QPushButton("▶ Preview")
        b_preview.clicked.connect(self._preview_tts)
        spl.addWidget(self._tts_speed_combo); spl.addWidget(b_preview); spl.addStretch()
        sec_tts.addRow("Speed", speed_row)

        from tts.engine import keep_recording_during_tts
        self._cb_keeprec = QCheckBox("Keep recording (no pause)")
        self._cb_keeprec.setChecked(True)
        keep_recording_during_tts.set()
        self._cb_keeprec.toggled.connect(
            lambda on: keep_recording_during_tts.set() if on else keep_recording_during_tts.clear()
        )
        sec_tts.addRow("During TTS", self._cb_keeprec)
        outer.addWidget(sec_tts)

        # ═══ Section: AI · Inline translate ════════════════════════════
        sec_ai = _Section("AI · Inline translate", "\U0001f9e0", expanded=False)
        lang_display = [f"{SUPPORTED_LANGS[c]['flag']} {SUPPORTED_LANGS[c]['name']}"
                        for c in self._lang_codes]
        self._cb_inline = QCheckBox("Enable  (Ctrl+Alt+T replace · Ctrl+Alt+D popup)")
        self._cb_inline.setChecked(bool(INLINE_ENABLED))
        self._cb_inline.toggled.connect(self._on_inline_toggle)
        sec_ai.addRow("Inline translate", self._cb_inline)

        pair_row = QWidget(); prl = QHBoxLayout(pair_row)
        prl.setContentsMargins(0, 0, 0, 0); prl.setSpacing(6)
        self._inline_from_combo = QComboBox(); self._inline_from_combo.addItems(lang_display)
        self._inline_to_combo = QComboBox(); self._inline_to_combo.addItems(lang_display)
        _from_idx = self._lang_codes.index(INLINE_FROM) if INLINE_FROM in self._lang_codes else 0
        _to_idx = self._lang_codes.index(INLINE_TO) if INLINE_TO in self._lang_codes else 0
        self._inline_from_combo.setCurrentIndex(_from_idx)
        self._inline_to_combo.setCurrentIndex(_to_idx)
        self._inline_from_combo.currentIndexChanged.connect(self._save_inline_cfg)
        self._inline_to_combo.currentIndexChanged.connect(self._save_inline_cfg)
        prl.addWidget(self._inline_from_combo)
        prl.addWidget(QLabel("→"))
        prl.addWidget(self._inline_to_combo)
        sec_ai.addRow("Inline pair", pair_row)

        self._inline_engine_combo = QComboBox()
        self._inline_engine_combo.addItems(["Qwen local", "Gemini"])
        self._inline_engine_combo.setCurrentText(
            INLINE_ENGINE if INLINE_ENGINE in ("Qwen local", "Gemini") else "Qwen local")
        self._inline_engine_combo.currentIndexChanged.connect(self._save_inline_cfg)
        sec_ai.addRow("Inline engine", self._inline_engine_combo)
        outer.addWidget(sec_ai)

        # ═══ Section: Recording ════════════════════════════════════════
        sec_rec = _Section("Recording", "\U0001f4c1", expanded=False)
        self._rec_dir_btn = QPushButton(self._record_dir or "Choose folder…")
        self._rec_dir_btn.setToolTip(self._record_dir or "")
        self._rec_dir_btn.clicked.connect(self._choose_record_dir)
        sec_rec.addRow("Folder", self._rec_dir_btn)
        self._rec_fmt_combo = QComboBox()
        self._rec_fmt_combo.addItems(["txt", "md", "srt"])
        sec_rec.addRow("Format", self._rec_fmt_combo)
        outer.addWidget(sec_rec)

        # ═══ Section: Display ══════════════════════════════════════════
        sec_disp = _Section("Display", "\U0001f5a5", expanded=False)
        font_row = QWidget(); fl = QHBoxLayout(font_row)
        fl.setContentsMargins(0, 0, 0, 0); fl.setSpacing(6)
        b_fdown = QPushButton("A-"); b_fdown.setFixedWidth(40); b_fdown.clicked.connect(self._font_down)
        b_fup = QPushButton("A+"); b_fup.setFixedWidth(40); b_fup.clicked.connect(self._font_up)
        fl.addWidget(b_fdown); fl.addWidget(b_fup); fl.addStretch()
        sec_disp.addRow("Font size", font_row)
        self.dual_btn = QPushButton("\U0001f4d6 Dual")
        self.dual_btn.clicked.connect(self._toggle_dual)
        sec_disp.addRow("View", self.dual_btn)
        b_terms = QPushButton("⚙ Edit glossary")
        b_terms.clicked.connect(self._open_terms_editor)
        sec_disp.addRow("Glossary", b_terms)
        outer.addWidget(sec_disp)

        # ═══ Section: System monitor ═══════════════════════════════════
        sec_mon = _Section("System monitor", "\U0001f4ca", expanded=True)
        self._cb_sysmon = QCheckBox("Show in status bar")
        self._cb_sysmon.setChecked(True)
        self._cb_sysmon.toggled.connect(self._on_sysmon_toggle)
        sec_mon.addRow("Monitor", self._cb_sysmon)

        def _mk_bar():
            b = QProgressBar()
            b.setRange(0, 100)
            b.setValue(0)
            b.setFixedHeight(15)
            return b

        self._pb_app = _mk_bar();  sec_mon.addRow("App RAM", self._pb_app)
        self._pb_sram = _mk_bar(); sec_mon.addRow("System RAM", self._pb_sram)
        self._pb_vram = _mk_bar(); sec_mon.addRow("VRAM", self._pb_vram)
        self._pb_gpu = _mk_bar();  sec_mon.addRow("GPU", self._pb_gpu)
        self._pb_cpu = _mk_bar();  sec_mon.addRow("CPU", self._pb_cpu)
        outer.addWidget(sec_mon)

        outer.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(280)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.hide()
        self._settings_dock = dock

    @staticmethod
    def _doc_empty(te):
        doc = te.document()
        return doc.blockCount() == 1 and not doc.firstBlock().text()

    def _is_near_bottom(self, te):
        sb = te.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def _scroll_end(self, te):
        sb = te.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_block(self, te, text, color):
        """Append text as a new paragraph block, return its block number."""
        at_bottom = self._is_near_bottom(te)
        cursor = te.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not self._doc_empty(te):
            cursor.insertBlock()
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text, fmt)
        block_no = cursor.blockNumber()
        if at_bottom:
            self._scroll_end(te)
        return block_no

    def _set_block_text(self, te, block_no, text, color):
        doc = te.document()
        block = doc.findBlockByNumber(block_no)
        if not block.isValid():
            self._append_block(te, text, color)
            return
        at_bottom = self._is_near_bottom(te)
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.StartOfBlock)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text, fmt)
        if at_bottom:
            self._scroll_end(te)

    def _append_colored(self, te, text, color, bold=False):
        at_bottom = self._is_near_bottom(te)
        cursor = te.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        if not self._doc_empty(te):
            cursor.insertText("\n", fmt)
        cursor.insertText(text, fmt)
        if at_bottom:
            self._scroll_end(te)

    # ── Font control ──────────────────────────────────────────────────
    def _font_up(self):
        self._font_size = min(self._font_size + 2, 40)
        self._apply_font()

    def _font_down(self):
        self._font_size = max(self._font_size - 2, 8)
        self._apply_font()

    def _apply_font(self):
        self.text_jp.setFont(QFont("Yu Gothic UI", self._font_size))
        self.text_vi.setFont(QFont("Segoe UI", self._font_size))

    # ── Language helpers ──────────────────────────────────────────────
    def _get_src_lang(self) -> str:
        idx = self._src_combo.currentIndex()
        return self._lang_codes[idx] if idx >= 0 else "ja"

    def _get_tgt_lang(self) -> str:
        idx = self._tgt_combo.currentIndex()
        return self._lang_codes[idx] if idx >= 0 else "vi"

    def _get_lang_pair(self) -> tuple[str, str]:
        return self._get_src_lang(), self._get_tgt_lang()

    def _update_panel_labels(self):
        src = SUPPORTED_LANGS.get(self._get_src_lang(), {})
        tgt = SUPPORTED_LANGS.get(self._get_tgt_lang(), {})
        self._src_label.setText(f"{src.get('flag','')} {src.get('name','Source')}")
        self._tgt_label.setText(f"{tgt.get('flag','')} {tgt.get('name','Target')}")

    def _on_lang_change(self, *_):
        self._update_panel_labels()
        self._refresh_voice_list()
        src = self._get_src_lang()
        tgt = self._get_tgt_lang()
        self.setWindowTitle(
            f"Interview STT — {SUPPORTED_LANGS[src]['name']} → {SUPPORTED_LANGS[tgt]['name']}"
        )

    # ── Dual / Single toggle ─────────────────────────────────────────
    def _toggle_dual(self):
        self._dual_mode = not self._dual_mode
        if self._dual_mode:
            self._vi_panel.setVisible(True)
            self.dual_btn.setText("\U0001f4d6 Dual")
        else:
            self._vi_panel.setVisible(False)
            self.dual_btn.setText("\U0001f4c4 Single")

    # ── AI Chat popup ────────────────────────────────────────────────
    def _open_chat(self):
        if self._chat_dialog is None:
            self._chat_dialog = ChatDialog(
                self,
                get_source_text=lambda: self.text_jp.toPlainText(),
                get_translation_text=lambda: self.text_vi.toPlainText(),
                get_qwen=self._get_chat_qwen,
            )
        self._chat_dialog.show()
        self._chat_dialog.raise_()
        self._chat_dialog.activateWindow()

    def _get_chat_qwen(self):
        if self._qwen_instance is None:
            if self.translator_thread and getattr(self.translator_thread, "_qwen", None):
                self._qwen_instance = self.translator_thread._qwen
            else:
                from translation.qwen import load_qwen_translator
                self._qwen_instance = load_qwen_translator()
        return self._qwen_instance

    # ── Mind map ─────────────────────────────────────────────────────
    def _make_mindmap(self):
        if getattr(self, "_map_busy", False):
            return
        content = self.text_jp.toPlainText().strip()
        if not content:
            self._flash_status("⚠️ No transcript to map")
            return
        from config import SUPPORTED_LANGS as _SL
        self._map_lang = self._get_tgt_lang()
        lang_name = _SL.get(self._map_lang, {}).get("name", "English")
        self._map_busy = True
        self.set_status("🗺 Generating mind map...")
        self._map_worker = MindmapWorker(content, lang_name, self._get_chat_qwen)
        self._map_worker.done.connect(self._show_mindmap)
        self._map_worker.error.connect(self._mindmap_error)
        self._map_worker.start()

    def _show_mindmap(self, code, explanations):
        self._map_busy = False
        self.set_status("🗺 Mind map ready")
        dlg = MindmapDialog(self, code, explanations,
                            default_dir=self._record_dir or "",
                            qwen_getter=self._get_chat_qwen,
                            lang_code=getattr(self, "_map_lang", "en"))
        dlg.show()

    def _mindmap_error(self, msg):
        self._map_busy = False
        self._flash_status(f"⚠️ Mind map: {msg}")

    # ── Inline translate ─────────────────────────────────────────────
    def _maybe_warm_gemini(self):
        """Pre-open the Gemini connection so the first inline call is fast."""
        if not self._cb_inline.isChecked():
            return
        if self._inline_engine_combo.currentText().startswith("Gemini"):
            from config import GEMINI_API_KEY, GEMINI_MODEL
            if GEMINI_API_KEY:
                from gemini_client import warm_gemini
                warm_gemini(GEMINI_API_KEY, GEMINI_MODEL)

    def _on_inline_toggle(self, on):
        if on:
            self._hotkey.start()
            self._hotkey2.start()
            self._flash_status("⌨️ Inline ON — Ctrl+Alt+T replace · Ctrl+Alt+D popup")
            self._maybe_warm_gemini()
        else:
            self._hotkey.stop()
            self._hotkey2.stop()
        from config import save_config_value
        save_config_value("inline_enabled", bool(on))

    def _save_inline_cfg(self, *_):
        from config import save_config_value
        save_config_value("inline_from", self._lang_codes[self._inline_from_combo.currentIndex()])
        save_config_value("inline_to", self._lang_codes[self._inline_to_combo.currentIndex()])
        save_config_value("inline_engine", self._inline_engine_combo.currentText())
        self._maybe_warm_gemini()

    def _inline_translate_text(self, text):
        """Translate `text` using the inline pair + engine. Runs on a worker thread."""
        src = self._lang_codes[self._inline_from_combo.currentIndex()]
        tgt = self._lang_codes[self._inline_to_combo.currentIndex()]
        engine = self._inline_engine_combo.currentText()
        if engine.startswith("Gemini"):
            from config import GEMINI_API_KEY, GEMINI_MODEL, LANG_NAMES_EN
            if not GEMINI_API_KEY:
                raise RuntimeError("No Gemini API key")
            from gemini_client import get_gemini_client, fast_config
            client = get_gemini_client(GEMINI_API_KEY)
            sn = LANG_NAMES_EN.get(src, src)
            tn = LANG_NAMES_EN.get(tgt, tgt)
            prompt = (f"Translate the following text from {sn} to {tn}. "
                      f"Reply with ONLY the translation, no notes or quotes.\n\n{text}")
            cfg = fast_config(GEMINI_MODEL)
            kwargs = {"config": cfg} if cfg else {}
            r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, **kwargs)
            return (getattr(r, "text", "") or "").strip()
        else:
            qwen = self._get_chat_qwen()
            if not qwen:
                raise RuntimeError("Qwen not available")
            return qwen.translate(text, src, tgt)

    # ── Translate toggle ──────────────────────────────────────────────
    def _on_translate_toggle(self, checked):
        self._translate_on = checked
        if self._translate_on:
            self._ensure_translator()

    def _on_tts_toggle(self, checked):
        self._tts_on = checked
        if self._tts_on:
            self._ensure_tts()
        elif self.tts_thread:
            self.tts_thread.stop()
            self.tts_thread = None

    def _get_tts_device(self) -> str | None:
        name = self._tts_dev_combo.currentText()
        if name and name != "(Default)":
            return name
        return None

    def _get_tts_voice(self) -> str | None:
        display = self._tts_voice_combo.currentText()
        voices = get_voices_for_lang(self._get_tgt_lang())
        for v in voices:
            gender = "♀" if v["Gender"] == "Female" else "♂"
            label = f"{v['ShortName'].split('-')[-1].replace('Neural','')} {gender}"
            if label == display:
                return v["ShortName"]
        return None

    def _get_tts_rate(self) -> str:
        label = self._tts_speed_combo.currentText()
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
        self._tts_voice_combo.clear()
        self._tts_voice_combo.addItems(labels or ["(None)"])
        if labels:
            default = TTS_DEFAULT_VOICE.get(lang, "")
            idx = next((i for i, v in enumerate(voices) if v["ShortName"] == default), 0)
            self._tts_voice_combo.setCurrentIndex(idx)

    def _preview_tts(self):
        if not HAS_TTS:
            self._flash_status("⚠️ edge-tts not installed!")
            return
        self._flash_status("\U0001f508 Playing preview...")
        threading.Thread(target=self._do_preview_tts, daemon=True).start()

    def _do_preview_tts(self):
        sample_texts = {
            "vi": "Xin chao, cam on ban da den phong van.",
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

    # ── Device ────────────────────────────────────────────────────────
    def _refresh_devices(self):
        mode = self._audio_mode_val()
        if mode == "mic":
            self.devices = list_mic_devices()
        else:
            self.devices = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or [f"(No {mode} found)"]
        self.dev_combo.clear()
        self.dev_combo.addItems(dev_names)
        if dev_names:
            self.dev_combo.setCurrentIndex(0)

    def _audio_mode_val(self) -> str:
        return "mic" if self._rb_mic.isChecked() else "loopback"

    def _selected_device_index(self):
        sel = self.dev_combo.currentText()
        for idx, name in self.devices:
            if name == sel:
                return idx
        return None

    def _load_diarization(self):
        load_diarization_pipeline()
        if _diar_mod.HAS_DIARIZATION:
            self.status_queue.put("Diarization OK. Listening...")

    # ── Start / Stop ──────────────────────────────────────────────────
    def toggle(self):
        if not self.running:
            self.start()
        else:
            self.stop()

    def _set_controls_enabled(self, enabled):
        self.dev_combo.setEnabled(enabled)
        self._src_combo.setEnabled(enabled)
        self._tgt_combo.setEnabled(enabled)
        self._engine_combo.setEnabled(enabled)
        self._chunk_combo.setEnabled(enabled)

    def start(self):
        self.running = True
        self.btn.setText("■ Stop")
        self.btn.setStyleSheet("background:#c62828; color:white; font-weight:bold;")
        self._set_controls_enabled(False)

        if not _diar_mod.HAS_DIARIZATION and HF_TOKEN:
            self.set_status("Loading diarization model...")
            threading.Thread(target=self._load_diarization, daemon=True).start()

        if self._audio_mode_val() == "mic":
            self.audio = AudioMic(self.frame_queue,
                                  device_index=self._selected_device_index())
        else:
            self.audio = AudioLoopback(self.frame_queue,
                                       device_index=self._selected_device_index())
        try:
            self.audio.start()
        except Exception as e:
            self.set_status(f"Audio error: {e}")
            self.running = False
            self.btn.setText("▶ Start")
            self.btn.setStyleSheet("background:#2e7d32; color:white; font-weight:bold;")
            self._set_controls_enabled(True)
            return

        engine = self._engine_combo.currentText()
        chunk_text = self._chunk_combo.currentText()
        chunk_sec = "auto" if chunk_text == "Auto" else float(chunk_text.replace("s", ""))
        if engine == "ReazonSpeech":
            self.transcriber = ReazonSpeechTranscriber(
                self.frame_queue, self.text_queue, self.status_queue,
                lambda: (self.audio.device_rate, self.audio.channels),
                get_src_lang=self._get_src_lang,
                chunk_sec=chunk_sec,
            )
        else:
            self.transcriber = Transcriber(
                self.frame_queue, self.text_queue, self.status_queue,
                lambda: (self.audio.device_rate, self.audio.channels),
                use_diarization=self._use_diarization,
                get_src_lang=self._get_src_lang,
                chunk_sec=chunk_sec,
            )
        self.transcriber.start()

        # Auto-start translator nếu dịch đang bật
        if self._translate_on:
            self._ensure_translator()

    def stop(self):
        self.running = False
        self.btn.setText("▶ Start")
        self.btn.setStyleSheet("background:#2e7d32; color:white; font-weight:bold;")
        self._set_controls_enabled(True)
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
        self._last_speaker = None
        _speaker_registry.reset()
        self.set_status("Stopped")

    # ── Clear / Copy ──────────────────────────────────────────────────
    def clear(self):
        self.text_jp.clear()
        self.text_vi.clear()
        self._pending_vi.clear()
        self._seg_counter = 0
        self._vi_next_id = 0
        self._last_speaker = None
        self._jp_last_char = ""

    def copy_jp(self):
        content = self.text_jp.toPlainText().strip()
        if content:
            QApplication.clipboard().setText(content)
            self._flash_status("✅ Copied source!")

    def copy_vi(self):
        content = self.text_vi.toPlainText().strip()
        if content:
            QApplication.clipboard().setText(content)
            self._flash_status("✅ Copied translation!")

    # ── AI automation (generic) ──────────────────────────────────────
    def _select_ai(self, name, cfg):
        self._ai_choice = name
        self._ai_btn.setText(f"\U0001f916 {name}")

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
        content = self.text_jp.toPlainText().strip()
        if not content:
            self._flash_status("⚠️ No text to send!")
            return
        QApplication.clipboard().setText(content)
        hwnd = open_fn()
        self._ai_target = (name, find_fn, click_fn)
        if hwnd:
            self._flash_status(f"⏳ Sending to {name}...")
            QTimer.singleShot(400, self._ai_click_and_paste)
        else:
            self._flash_status(f"\U0001f680 Opening {name}...")
            QTimer.singleShot(launch_delay, self._ai_click_and_paste)

    def _ai_click_and_paste(self):
        name, find_fn, click_fn = self._ai_target
        hwnd = find_fn()
        if hwnd:
            click_fn(hwnd)
        QTimer.singleShot(350, self._ai_do_paste)

    def _ai_do_paste(self):
        user32 = ctypes.windll.user32
        KEYUP = 0x0002
        user32.keybd_event(0x11, 0, 0, 0)
        user32.keybd_event(0x56, 0, 0, 0)
        user32.keybd_event(0x56, 0, KEYUP, 0)
        user32.keybd_event(0x11, 0, KEYUP, 0)
        QTimer.singleShot(300, self._ai_do_enter)

    def _ai_do_enter(self):
        user32 = ctypes.windll.user32
        KEYUP = 0x0002
        user32.keybd_event(0x0D, 0, 0, 0)
        user32.keybd_event(0x0D, 0, KEYUP, 0)
        name = self._ai_target[0]
        self._flash_status(f"✅ Sent! {name} is responding...")

    # ── Status helpers ────────────────────────────────────────────────
    def _flash_status(self, msg):
        old = self.status.text()
        self.status.setText(msg)
        self.status.setStyleSheet("color:#00e676; font-size:11px;")
        QTimer.singleShot(
            2000,
            lambda: (self.status.setText(old),
                     self.status.setStyleSheet("color:#555; font-size:11px;")),
        )

    def set_status(self, msg):
        self.status.setText(msg)
        self.status.setStyleSheet("color:#555; font-size:11px;")

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

            at_bottom = self._is_near_bottom(self.text_jp)
            cursor = self.text_jp.textCursor()
            cursor.movePosition(QTextCursor.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))

            same_speaker = (speaker == self._last_speaker)
            if not same_speaker:
                if self._last_speaker is not None:
                    cursor.insertText("\n", fmt)
                if speaker:
                    cursor.insertText(f"{speaker}\n", fmt)
                cursor.insertText(jp_text, fmt)
                self._last_speaker = speaker
            else:
                if self._jp_last_char in ("。", "！", "？", ".", "!", "?"):
                    cursor.insertText("\n" + jp_text, fmt)
                else:
                    cursor.insertText(jp_text, fmt)

            self._jp_last_char = jp_text[-1] if jp_text else self._jp_last_char
            if at_bottom:
                self._scroll_end(self.text_jp)

            # Live record of the source transcript
            if self._recorder:
                self._recorder.write_source(jp_text, speaker)

            # FIFO: gửi (seg_id, jp_text) qua translator
            if self._translate_on:
                seg_id = self._seg_counter
                self._seg_counter += 1
                self.jp_trans_queue.put((seg_id, jp_text))
                # Hiện placeholder trong panel VI
                block_no = self._append_block(self.text_vi, "⏳ ...", "#555")
                self._pending_vi[seg_id] = block_no

        # VI text từ Qwen / Google — FIFO thay thế placeholder
        while not self.vi_queue.empty():
            result = self.vi_queue.get()
            if isinstance(result, tuple):
                seg_id, vi_text = result
            else:
                seg_id, vi_text = None, result

            if seg_id is not None and seg_id in self._pending_vi:
                block_no = self._pending_vi.pop(seg_id)
                self._set_block_text(self.text_vi, block_no, vi_text, "#aff3c3")
            else:
                self._append_block(self.text_vi, vi_text, "#aff3c3")

            if self._tts_on and vi_text.strip():
                self.tts_queue.put(vi_text)

        # Lỗi dịch
        while not self.err_queue.empty():
            err = self.err_queue.get()
            self.set_status(f"⚠️ {err}")

    # ── Terms Editor ────────────────────────────────────────────────
    def _open_terms_editor(self):
        import translation.terms as terms_mod

        win = QDialog(self)
        win.setWindowTitle("⚙ Custom Terms Editor")
        win.resize(500, 420)
        win.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        win.setStyleSheet(DARK_QSS)
        lay = QVBoxLayout(win)

        hint = QLabel("One per line: Source = Target")
        hint.setStyleSheet("color:#888;")
        lay.addWidget(hint)

        txt = QTextEdit()
        txt.setStyleSheet("background:#111; color:#eee;")
        txt.setFont(QFont("Segoe UI", 12))
        lay.addWidget(txt)

        # Load terms hiện tại
        for jp, vi in terms_mod._custom_terms.items():
            txt.append(f"{jp} = {vi}")

        status_lbl = QLabel("")
        status_lbl.setStyleSheet("color:#888;")
        lay.addWidget(status_lbl)

        def save():
            content = txt.toPlainText().strip()
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
                status_lbl.setText(f"❌ Save error: {e}")
                status_lbl.setStyleSheet("color:#ef5350;")
                return
            # Cập nhật hint trong Qwen translator nếu đang chạy
            if (self.translator_thread and hasattr(self.translator_thread, '_qwen')
                    and self.translator_thread._qwen):
                self.translator_thread._qwen._terms_hint = _build_terms_hint()
            status_lbl.setText(f"✅ Saved {len(new_terms)} terms!")
            status_lbl.setStyleSheet("color:#3fb950;")
            QTimer.singleShot(
                2000,
                lambda: (status_lbl.setText(""), status_lbl.setStyleSheet("color:#888;")),
            )

        btn_bar = QHBoxLayout()
        lay.addLayout(btn_bar)
        b_save = QPushButton("\U0001f4be Save")
        b_save.setStyleSheet("background:#238636; color:white; font-weight:bold;")
        b_save.clicked.connect(save)
        b_close = QPushButton("Close")
        b_close.clicked.connect(win.close)
        path_lbl = QLabel(f"\U0001f4c1 {TERMS_PATH}")
        path_lbl.setStyleSheet("color:#555; font-size:10px;")
        btn_bar.addWidget(b_save)
        btn_bar.addWidget(b_close)
        btn_bar.addStretch()
        btn_bar.addWidget(path_lbl)

        win.exec()

    def closeEvent(self, event):
        self.stop()
        if self.translator_thread:
            self.translator_thread.stop()
        if self.tts_thread:
            self.tts_thread.stop()
        if self._recorder:
            self._recorder.close()
        if getattr(self, "_hotkey", None):
            self._hotkey.stop()
        if getattr(self, "_hotkey2", None):
            self._hotkey2.stop()
        if getattr(self, "_sysmon", None):
            self._sysmon.stop()
        event.accept()
