# -*- coding: utf-8 -*-
"""
AI Chat popup — chat with Gemini (cloud) or local Qwen.
Can insert the live transcript (source or translation) as context.
"""

import re
import threading

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
    QTextEdit, QPlainTextEdit, QToolButton, QMenu, QCheckBox,
)
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QKeyEvent
from PySide6.QtCore import Qt, QObject, Signal

from config import GEMINI_API_KEY, GEMINI_MODEL


CHAT_QSS = """
QDialog { background:#0d1117; color:#c9d1d9; }
QComboBox { background:#1c2128; border:1px solid #30363d; border-radius:6px; padding:4px 8px; color:#c9d1d9; }
QComboBox QAbstractItemView { background:#161b22; color:#c9d1d9; border:1px solid #30363d; selection-background-color:#1f6feb; }
QPushButton { background:#21262d; border:1px solid #30363d; border-radius:6px; padding:6px 12px; color:#c9d1d9; }
QPushButton:hover { background:#30363d; }
QPushButton#send { background:#1f6feb; border:none; color:white; font-weight:500; }
QPushButton#send:hover { background:#2b7bf3; }
QPushButton#send:disabled { background:#21262d; color:#555; }
QToolButton { background:#21262d; border:1px solid #30363d; border-radius:6px; padding:6px 10px; color:#c9d1d9; }
QToolButton:hover { background:#30363d; }
QToolButton::menu-indicator { image:none; }
QMenu { background:#1c2128; color:#c9d1d9; border:1px solid #30363d; border-radius:8px; padding:4px; }
QMenu::item { padding:6px 16px; border-radius:5px; }
QMenu::item:selected { background:#1f6feb; color:white; }
QTextEdit, QPlainTextEdit { background:#0a0c10; color:#e6edf3; border:1px solid #1c2128; border-radius:8px; padding:8px; }
QScrollBar:vertical { background:transparent; width:9px; margin:2px; }
QScrollBar::handle:vertical { background:#30363d; border-radius:4px; min-height:24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QLabel { color:#8b949e; }
"""


class ChatWorker(QObject):
    """Runs one chat turn in a background thread, streams text back via signals."""
    chunk = Signal(str)
    done = Signal()
    error = Signal(str)

    def __init__(self, engine, history, model, qwen_getter, web_search=True):
        super().__init__()
        self._engine = engine
        self._history = history          # list of {"role": "user"|"model", "text": str}
        self._model = model
        self._qwen_getter = qwen_getter
        self._web = web_search
        self._stop = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop = True

    def _run(self):
        try:
            if self._engine == "Gemini":
                self._run_gemini()
            else:
                self._run_local()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()

    def _run_gemini(self):
        if not GEMINI_API_KEY:
            self.error.emit("No Gemini API key in config.json")
            return
        from google.genai import types
        from gemini_client import get_gemini_client, chat_config

        client = get_gemini_client(GEMINI_API_KEY)
        contents = [
            types.Content(role=h["role"], parts=[types.Part(text=h["text"])])
            for h in self._history
        ]
        cfg = chat_config(self._model, self._web)
        kwargs = {"config": cfg} if cfg else {}
        stream = client.models.generate_content_stream(
            model=self._model, contents=contents, **kwargs,
        )
        for ch in stream:
            if self._stop:
                break
            t = getattr(ch, "text", None)
            if t:
                self.chunk.emit(t)

    def _run_local(self):
        qwen = self._qwen_getter()
        if not qwen:
            self.error.emit("Local Qwen model not available")
            return
        import torch

        msgs = [
            {"role": "assistant" if h["role"] == "model" else "user",
             "content": h["text"]}
            for h in self._history
        ]
        prompt = qwen.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = qwen.tokenizer(prompt, return_tensors="pt").to(qwen._device)
        with torch.no_grad():
            out = qwen.model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                temperature=None, top_p=None,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = qwen.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text:
            self.chunk.emit(text)


class _InputBox(QPlainTextEdit):
    """Multi-line input: Enter sends, Shift+Enter newline."""
    submitted = Signal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class ChatDialog(QDialog):
    def __init__(self, parent, get_source_text, get_translation_text, get_qwen):
        super().__init__(parent)
        self._get_source = get_source_text
        self._get_translation = get_translation_text
        self._get_qwen = get_qwen
        self._history = []
        self._worker = None
        self._streaming = False

        self.setWindowTitle("🧠 AI Chat")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.resize(520, 600)
        self.setStyleSheet(CHAT_QSS)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── Top: engine selector ──
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("Engine"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(["Gemini", "Local Qwen"])
        self._engine_combo.setMinimumWidth(120)
        top.addWidget(self._engine_combo)
        self._model_lbl = QLabel(GEMINI_MODEL)
        self._model_lbl.setStyleSheet("color:#6e7681; font-size:11px;")
        top.addWidget(self._model_lbl)
        top.addStretch()
        self._web_cb = QCheckBox("🌐 Web")
        self._web_cb.setChecked(True)
        self._web_cb.setToolTip("Let Gemini search the web for fresh, factual answers")
        top.addWidget(self._web_cb)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear_chat)
        top.addWidget(b_clear)
        lay.addLayout(top)
        self._engine_combo.currentTextChanged.connect(self._on_engine_change)

        # ── Chat view ──
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(QFont("Segoe UI", 11))
        lay.addWidget(self.view, 1)

        # ── Insert transcript + input ──
        in_row = QHBoxLayout()
        in_row.setSpacing(6)

        self._insert_btn = QToolButton()
        self._insert_btn.setText("＋ Transcript")
        self._insert_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self._insert_btn)
        menu.addAction("Insert source").triggered.connect(
            lambda: self._insert_transcript("source"))
        menu.addAction("Insert translation").triggered.connect(
            lambda: self._insert_transcript("translation"))
        self._insert_btn.setMenu(menu)
        in_row.addWidget(self._insert_btn)
        in_row.addStretch()
        lay.addLayout(in_row)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.input = _InputBox()
        self.input.setPlaceholderText("Type a message…  (Enter = send, Shift+Enter = newline)")
        self.input.setFixedHeight(64)
        self.input.submitted.connect(self._send)
        bottom.addWidget(self.input, 1)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("send")
        self.send_btn.setFixedWidth(72)
        self.send_btn.clicked.connect(self._send)
        bottom.addWidget(self.send_btn)
        lay.addLayout(bottom)

        self._append_system("Ready. Ask anything, or insert the transcript as context.")

    # ── Engine ──
    def _on_engine_change(self, name):
        if name == "Gemini":
            self._model_lbl.setText(GEMINI_MODEL)
        else:
            self._model_lbl.setText("Qwen 3 1.7B (local GPU)")

    # ── Transcript insert ──
    def _insert_transcript(self, which):
        text = (self._get_source() if which == "source" else self._get_translation()).strip()
        if not text:
            self._append_system(f"No {which} transcript yet.")
            return
        prefix = "Source transcript" if which == "source" else "Translation"
        cur = self.input.toPlainText()
        block = f"[{prefix}]\n{text}\n"
        self.input.setPlainText((cur + "\n" + block) if cur.strip() else block)
        self.input.moveCursor(QTextCursor.End)

    # ── Send / receive ──
    def _send(self):
        if self._streaming:
            return
        msg = self.input.toPlainText().strip()
        if not msg:
            return
        self.input.clear()
        self._history.append({"role": "user", "text": msg})
        self._append_bubble("You", "#58a6ff", msg)

        engine = self._engine_combo.currentText()
        bot_name = "Gemini" if engine == "Gemini" else "Qwen"
        bot_color = "#a5d6ff" if engine == "Gemini" else "#d2a8ff"
        self._start_bot_bubble(bot_name, bot_color)
        self._bot_accum = ""

        self._streaming = True
        self.send_btn.setEnabled(False)
        self.send_btn.setText("…")

        self._worker = ChatWorker(
            engine="Gemini" if engine == "Gemini" else "Local",
            history=list(self._history),
            model=GEMINI_MODEL,
            qwen_getter=self._get_qwen,
            web_search=self._web_cb.isChecked(),
        )
        self._worker.chunk.connect(self._on_chunk)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_chunk(self, text):
        self._bot_accum += text
        self._append_stream(text)

    def _on_error(self, msg):
        self._append_stream(f"\n⚠️ {msg}")
        self._bot_accum += f"\n⚠️ {msg}"

    def _on_done(self):
        self._streaming = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        if self._bot_accum.strip():
            self._history.append({"role": "model", "text": self._bot_accum})

    # ── Rendering ──
    def _near_bottom(self):
        sb = self.view.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def _scroll_end(self):
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_system(self, text):
        at_bottom = self._near_bottom()
        cur = self.view.textCursor(); cur.movePosition(QTextCursor.End)
        fmt = QTextCharFormat(); fmt.setForeground(QColor("#6e7681"))
        fmt.setFontItalic(True)
        if self.view.document().characterCount() > 1:
            cur.insertText("\n", fmt)
        cur.insertText(text, fmt)
        if at_bottom:
            self._scroll_end()

    def _append_bubble(self, name, color, text):
        at_bottom = self._near_bottom()
        cur = self.view.textCursor(); cur.movePosition(QTextCursor.End)
        name_fmt = QTextCharFormat(); name_fmt.setForeground(QColor(color))
        name_fmt.setFontWeight(QFont.Bold)
        body_fmt = QTextCharFormat(); body_fmt.setForeground(QColor("#e6edf3"))
        if self.view.document().characterCount() > 1:
            cur.insertText("\n\n", body_fmt)
        cur.insertText(f"{name}\n", name_fmt)
        cur.insertText(text, body_fmt)
        if at_bottom:
            self._scroll_end()

    def _start_bot_bubble(self, name, color):
        cur = self.view.textCursor(); cur.movePosition(QTextCursor.End)
        name_fmt = QTextCharFormat(); name_fmt.setForeground(QColor(color))
        name_fmt.setFontWeight(QFont.Bold)
        if self.view.document().characterCount() > 1:
            cur.insertText("\n\n", name_fmt)
        cur.insertText(f"{name}\n", name_fmt)
        self._scroll_end()

    def _append_stream(self, text):
        at_bottom = self._near_bottom()
        cur = self.view.textCursor(); cur.movePosition(QTextCursor.End)
        fmt = QTextCharFormat(); fmt.setForeground(QColor("#e6edf3"))
        cur.insertText(text, fmt)
        if at_bottom:
            self._scroll_end()

    def _clear_chat(self):
        self._history.clear()
        self.view.clear()
        self._append_system("Chat cleared.")
