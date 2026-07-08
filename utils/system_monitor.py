# -*- coding: utf-8 -*-
"""
SystemMonitor — reads RAM / VRAM / CPU / GPU usage on a background thread and
emits them to the UI. Uses psutil (RAM/CPU) and pynvml (VRAM/GPU); both are
optional — missing metrics are simply omitted.
"""

import threading

from PySide6.QtCore import QObject, Signal


class SystemMonitor(QObject):
    updated = Signal(dict)

    def __init__(self, interval: float = 2.0, parent=None):
        super().__init__(parent)
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        # ── psutil (RAM / CPU) ──
        try:
            import psutil
            proc = psutil.Process()
            proc.cpu_percent(None)        # prime
            psutil.cpu_percent(None)      # prime
            ncpu = psutil.cpu_count() or 1
        except Exception:
            psutil = None; proc = None; ncpu = 1

        # ── pynvml (VRAM / GPU) ──
        nvml = None; handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nvml = pynvml
        except Exception:
            nvml = None

        while not self._stop.is_set():
            s = {}
            if psutil:
                try:
                    vm = psutil.virtual_memory()
                    s["app_ram_gb"] = proc.memory_info().rss / 1e9
                    s["sys_ram_used_gb"] = vm.used / 1e9
                    s["sys_ram_total_gb"] = vm.total / 1e9
                    s["sys_ram_pct"] = float(vm.percent)
                    s["cpu_total_pct"] = float(psutil.cpu_percent(None))
                    s["cpu_app_pct"] = min(100.0, proc.cpu_percent(None) / ncpu)
                except Exception:
                    pass
            if nvml and handle:
                try:
                    mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    s["vram_used_gb"] = mem.used / 1e9
                    s["vram_total_gb"] = mem.total / 1e9
                    s["vram_pct"] = (mem.used / mem.total * 100) if mem.total else 0.0
                    s["gpu_util_pct"] = float(util.gpu)
                    name = nvml.nvmlDeviceGetName(handle)
                    s["gpu_name"] = name.decode() if isinstance(name, bytes) else name
                except Exception:
                    pass
            self.updated.emit(s)
            self._stop.wait(self._interval)
