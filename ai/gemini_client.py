# -*- coding: utf-8 -*-
"""
Cached Gemini client — reuse one google-genai Client (and its HTTP connection
pool) across calls instead of recreating it every time. Big latency win for
repeated inline-translate / chat requests.
"""

import threading

_clients = {}
_lock = threading.Lock()


def get_gemini_client(api_key: str):
    """Return a cached genai.Client for this api_key (creates on first use)."""
    with _lock:
        c = _clients.get(api_key)
        if c is None:
            from google import genai
            c = genai.Client(api_key=api_key)
            _clients[api_key] = c
        return c


def fast_config(model: str):
    """Config that disables 'thinking' for low latency on 2.5 models.
    Translation/short replies don't need reasoning → much faster."""
    try:
        from google.genai import types
        if "2.5" in (model or ""):
            return types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
    except Exception:
        pass
    return None


def chat_config(model: str, web_search: bool = True):
    """Config for chat: enable Google Search grounding (fresh, factual answers).
    Thinking is left ON (default) for depth — chat isn't latency-critical."""
    try:
        from google.genai import types
        tools = []
        if web_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if tools:
            return types.GenerateContentConfig(tools=tools)
    except Exception:
        pass
    return None


def warm_gemini(api_key: str, model: str = "gemini-2.5-flash"):
    """Open the connection ahead of time so the first real call is fast.
    Best-effort: errors (expired key / offline) are ignored."""
    def _run():
        try:
            c = get_gemini_client(api_key)
            c.models.generate_content(model=model, contents="hi")
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()
