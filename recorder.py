# -*- coding: utf-8 -*-
"""
RecordWriter — live-append the source transcript to a file (txt / md / srt).
Does NOT record audio; saves text only.
"""

import os
import time
import datetime
import threading


class RecordWriter:
    def __init__(self, folder: str, fmt: str = "txt", info: str = ""):
        self.fmt = fmt if fmt in ("txt", "md", "srt") else "txt"
        self.folder = folder
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(folder, f"meeting_{ts}.{self.fmt}")
        self._start = time.time()
        self._idx = 0
        self._lock = threading.Lock()
        self._f = open(self.path, "w", encoding="utf-8")
        self._write_header(info)

    def _write_header(self, info):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if self.fmt == "md":
            self._f.write(f"# Meeting transcript — {now}\n\n")
            if info:
                self._f.write(f"_{info}_\n\n")
        elif self.fmt == "txt":
            self._f.write(f"Meeting transcript — {now}\n")
            if info:
                self._f.write(f"{info}\n")
            self._f.write("=" * 44 + "\n\n")
        # srt: no header
        self._f.flush()

    @staticmethod
    def _srt_ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def write_source(self, text: str, speaker: str | None = None):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            if self.fmt == "srt":
                self._idx += 1
                start = time.time() - self._start
                dur = max(1.5, min(len(text) * 0.18, 8.0))
                end = start + dur
                who = f"[{speaker}] " if speaker else ""
                self._f.write(
                    f"{self._idx}\n"
                    f"{self._srt_ts(start)} --> {self._srt_ts(end)}\n"
                    f"{who}{text}\n\n"
                )
            else:
                clock = datetime.datetime.now().strftime("%H:%M:%S")
                who = f"{speaker}: " if speaker else ""
                if self.fmt == "md":
                    self._f.write(f"- **[{clock}]** {who}{text}\n")
                else:  # txt
                    self._f.write(f"[{clock}] {who}{text}\n")
            self._f.flush()

    def close(self):
        with self._lock:
            try:
                self._f.close()
            except Exception:
                pass
