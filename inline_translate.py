# -*- coding: utf-8 -*-
"""
Inline translate — system-wide hotkey that replaces the selected text in any
app with its translation, in place (Ctrl+C → translate → Ctrl+V).

GlobalHotkey  : Win32 RegisterHotKey in a dedicated thread with its own message
                loop (robust, no native event filter needed).
InlineTranslator : orchestrates the clipboard dance on the GUI thread, running
                   the heavy translate call on a worker thread.
"""

import ctypes
import threading
from ctypes import wintypes

from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QFrame


# Win32 constants
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
VK_CONTROL = 0x11
KEYEVENTF_KEYUP = 0x0002


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND), ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD), ("pt", _POINT),
    ]


def _release_modifiers():
    """Release any physically-held modifiers (the hotkey's Ctrl+Alt are still down)."""
    user32 = ctypes.windll.user32
    for vk in (0x12, 0x11, 0x10, 0x5B, 0x5C):  # Alt, Ctrl, Shift, LWin, RWin
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def send_ctrl_key(vk: int):
    """Send a clean Ctrl+<vk> to the foreground app (clears held modifiers first)."""
    user32 = ctypes.windll.user32
    _release_modifiers()
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


class GlobalHotkey(QObject):
    """System-wide hotkey. Emits `activated` on press (delivered to GUI thread)."""
    activated = Signal()
    failed = Signal(str)

    def __init__(self, mods: int, vk: int, parent=None):
        super().__init__(parent)
        self._mods = mods
        self._vk = vk
        self._tid = None
        self._thread = None
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._tid:
            user32 = ctypes.windll.user32
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        self._running = False
        self._tid = None

    def _run(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._tid = kernel32.GetCurrentThreadId()

        if not user32.RegisterHotKey(None, 1, self._mods | MOD_NOREPEAT, self._vk):
            self.failed.emit("RegisterHotKey failed — phím tắt có thể đang bị app khác chiếm")
            self._tid = None
            self._running = False
            return

        msg = _MSG()
        try:
            while True:
                r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r in (0, -1):          # WM_QUIT or error
                    break
                if msg.message == WM_HOTKEY:
                    self.activated.emit()
        finally:
            user32.UnregisterHotKey(None, 1)


class ResultPopup(QWidget):
    """Small floating box that shows the translation near the cursor."""
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setMaximumWidth(380)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        frame = QFrame(); frame.setObjectName("pop")
        frame.setStyleSheet(
            "#pop{background:#161b22;border:1px solid #30363d;border-radius:10px;}")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(12, 10, 12, 10); fl.setSpacing(5)
        self._src = QLabel(); self._src.setWordWrap(True)
        self._src.setStyleSheet("color:#6e7681; font-size:11px;")
        self._txt = QLabel(); self._txt.setWordWrap(True)
        self._txt.setStyleSheet("color:#e6edf3; font-size:14px;")
        self._txt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fl.addWidget(self._src); fl.addWidget(self._txt)
        lay.addWidget(frame)

        self._timer = QTimer(self); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_text(self, src, text):
        src = src.strip()
        self._src.setText(src[:120] + ("…" if len(src) > 120 else ""))
        self._txt.setText(text)
        self.adjustSize()
        self._reposition()
        self.show(); self.raise_()
        self._timer.start(12000)             # auto-dismiss after 12s

    def _reposition(self):
        pos = QCursor.pos()
        x, y = pos.x() + 14, pos.y() + 18
        scr = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        g = scr.availableGeometry()
        w, h = self.width(), self.height()
        if x + w > g.right():
            x = g.right() - w - 4
        if y + h > g.bottom():
            y = pos.y() - h - 12
        self.move(max(g.left() + 4, x), max(g.top() + 4, y))

    def mousePressEvent(self, e):
        self.hide()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()


class InlineTranslator(QObject):
    """Translate the selected text — replace it in place, or show a popup."""
    status = Signal(str)
    _translated = Signal()

    def __init__(self, translate_fn, parent=None):
        super().__init__(parent)
        self._translate_fn = translate_fn   # callable(text) -> str | None
        self._busy = False
        self._old = ""
        self._sel = ""
        self._result = None
        self._mode = "replace"
        self._popup = None
        self._translated.connect(self._after_translate)

    def trigger(self, mode="replace"):
        if self._busy:
            return
        self._busy = True
        self._mode = mode
        cb = QApplication.instance().clipboard()
        self._old = cb.text()
        _release_modifiers()              # let go of the still-held Ctrl+Alt from the hotkey
        QTimer.singleShot(90, self._do_copy)

    def _do_copy(self):
        send_ctrl_key(0x43)               # Ctrl+C (copy selection)
        QTimer.singleShot(180, self._after_copy)

    def _after_copy(self):
        cb = QApplication.instance().clipboard()
        sel = cb.text()
        if not sel.strip() or sel == self._old:
            self.status.emit("⚠️ Inline: hãy bôi đen đoạn cần dịch")
            self._busy = False
            return
        self._sel = sel
        self.status.emit("⏳ Inline translating...")

        def work():
            try:
                self._result = self._translate_fn(self._sel)
            except Exception as e:
                self._result = None
                self.status.emit(f"⚠️ Inline error: {e}")
            self._translated.emit()

        threading.Thread(target=work, daemon=True).start()

    def _after_translate(self):
        out = self._result
        self._result = None
        if not out or not out.strip():
            self.status.emit("⚠️ Inline: bản dịch trống")
            self._restore_clip()
            self._busy = False
            return
        if self._mode == "popup":
            self._show_popup(self._sel, out)
            self._restore_clip()          # don't touch the text — just restore clipboard
            self._busy = False
            self.status.emit("✅ Translated")
        else:
            cb = QApplication.instance().clipboard()
            cb.setText(out)
            send_ctrl_key(0x56)           # Ctrl+V (paste over selection)
            QTimer.singleShot(160, self._restore)

    def _show_popup(self, src, text):
        if self._popup is None:
            self._popup = ResultPopup()
        self._popup.show_text(src, text)

    def _restore_clip(self):
        QApplication.instance().clipboard().setText(self._old)

    def _restore(self):
        self._restore_clip()
        self.status.emit("✅ Inline translate done")
        self._busy = False
