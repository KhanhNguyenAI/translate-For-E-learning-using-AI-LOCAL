# -*- coding: utf-8 -*-
"""
Interview STT - Realtime Japanese speech-to-text from speaker (WASAPI loopback)
Chạy: python main.py
"""

import os
import sys

# Fix Windows: tìm đúng vị trí NVIDIA DLLs bằng importlib
# MUST run BEFORE any torch/cuda imports
if sys.platform == "win32":
    import importlib.util, pathlib
    for _pkg in ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime"]:
        _spec = importlib.util.find_spec(_pkg)
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                _bin = pathlib.Path(_loc) / "bin"
                if _bin.is_dir():
                    os.add_dll_directory(str(_bin))
                    print(f"[DLL] Added: {_bin}")

import tkinter as tk
from app import App

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
