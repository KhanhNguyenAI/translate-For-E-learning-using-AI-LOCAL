# -*- coding: utf-8 -*-
"""
Win32 AI automation — Copilot, Claude Desktop, ChatGPT.
Uses ctypes built-in (no external deps).
"""

import time
import ctypes
import ctypes.wintypes
import subprocess

# -------- Microsoft Copilot --------
COPILOT_APP_IDS = [
    "Microsoft.Windows.Ai.Copilot.Provider_8wekyb3d8bbwe!App",
    "MicrosoftWindows.Client.WebExperience_cw5n1h2txyewy!Copilot",
    "Microsoft.Copilot_8wekyb3d8bbwe!App",
]


def find_copilot_hwnd():
    """Tìm handle cửa sổ chính của Microsoft Copilot."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def _cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.lower()
                if "copilot" in title:
                    found.append(hwnd)
        return True

    user32.EnumWindows(_cb, 0)
    return found[0] if found else None


def open_or_focus_copilot():
    """Bring Microsoft Copilot lên foreground. Trả về hwnd nếu đang mở, None nếu vừa launch."""
    user32 = ctypes.windll.user32
    hwnd = find_copilot_hwnd()
    if hwnd:
        user32.ShowWindow(hwnd, 9)        # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        return hwnd

    # Thử mở bằng Windows Store app ID
    opened = False
    for app_id in COPILOT_APP_IDS:
        try:
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
            opened = True
            break
        except Exception:
            continue

    # Fallback: dùng phím Win+C (mở Copilot sidebar Windows 11)
    if not opened:
        VK_LWIN = 0x5B
        VK_C    = 0x43
        KEYUP   = 0x0002
        user32.keybd_event(VK_LWIN, 0, 0,    0)
        user32.keybd_event(VK_C,    0, 0,    0)
        user32.keybd_event(VK_C,    0, KEYUP, 0)
        user32.keybd_event(VK_LWIN, 0, KEYUP, 0)

    return None


def click_copilot_input(hwnd):
    """Click vào ô 'Message Copilot' (bottom-center, ~88% chiều cao window)."""
    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))

    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    x = rect.left + win_w // 2
    y = rect.top  + int(win_h * 0.88)

    user32.SetCursorPos(x, y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)   # MOUSEEVENTF_LEFTDOWN
    user32.mouse_event(0x0004, 0, 0, 0, 0)   # MOUSEEVENTF_LEFTUP


# -------- Claude Desktop --------
CLAUDE_APP_IDS = [
    "Anthropic.Claude_4mxp67smjv6yp!App",
    "Claude_4mxp67smjv6yp!App",
]


def find_claude_hwnd():
    """Tìm handle cửa sổ chính của Claude Desktop."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def _cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if "Claude" in title and "Code" not in title:
                    found.append(hwnd)
        return True

    user32.EnumWindows(_cb, 0)
    return found[0] if found else None


def open_or_focus_claude():
    """Bring Claude Desktop lên foreground. Trả về hwnd nếu đang mở, None nếu vừa launch."""
    user32 = ctypes.windll.user32
    hwnd = find_claude_hwnd()
    if hwnd:
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        return hwnd

    for app_id in CLAUDE_APP_IDS:
        try:
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
            return None
        except Exception:
            continue

    # Fallback: thử mở bằng tên exe
    try:
        subprocess.Popen(["claude.exe"])
    except Exception:
        pass
    return None


def click_claude_input(hwnd):
    """Click vào ô nhập chat của Claude Desktop (bottom-center, ~92% chiều cao)."""
    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))

    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    x = rect.left + win_w // 2
    y = rect.top  + int(win_h * 0.92)

    user32.SetCursorPos(x, y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)


# -------- ChatGPT --------
CHATGPT_APP_IDS = [
    "OpenAI.ChatGPT_2p2nf5s2dxmpy!App",
    "OpenAI.ChatGPT_8wekyb3d8bbwe!App",
]


def find_chatgpt_hwnd():
    """Tìm handle cửa sổ chính của ChatGPT."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def _cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if "ChatGPT" in title:
                    found.append(hwnd)
        return True

    user32.EnumWindows(_cb, 0)
    return found[0] if found else None


def open_or_focus_chatgpt():
    """Bring ChatGPT lên foreground."""
    user32 = ctypes.windll.user32
    hwnd = find_chatgpt_hwnd()
    if hwnd:
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        return hwnd

    for app_id in CHATGPT_APP_IDS:
        try:
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
            return None
        except Exception:
            continue
    return None


def click_chatgpt_input(hwnd):
    """Click vào ô nhập chat của ChatGPT (bottom-center, ~90% chiều cao)."""
    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))

    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    x = rect.left + win_w // 2
    y = rect.top  + int(win_h * 0.90)

    user32.SetCursorPos(x, y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)


# ── AI registry and options ─────────────────────────────────────────
AI_OPTIONS = {
    "Copilot":  {"color": "#0078d4", "icon": "\U0001f7e6"},
    "Claude":   {"color": "#d97706", "icon": "\U0001f7e7"},
    "ChatGPT":  {"color": "#10a37f", "icon": "\U0001f7e9"},
}

_AI_REGISTRY = {
    "Copilot":  (open_or_focus_copilot, find_copilot_hwnd, click_copilot_input, 4500),
    "Claude":   (open_or_focus_claude,  find_claude_hwnd,  click_claude_input,  5000),
    "ChatGPT":  (open_or_focus_chatgpt, find_chatgpt_hwnd, click_chatgpt_input, 4500),
}
