# Interview STT — Project Context for AI Agents

## Overview
Real-time Speech-to-Text + Translation + AI Analysis app for online interviews (Teams, Zoom, Meet). Fully local GPU-accelerated, zero cloud cost.

**Stack**: Python 3.10+, PySide6, faster-whisper, Qwen 3 1.7B, edge-tts, pyannote.audio  
**OS**: Windows 10/11 only (WASAPI loopback, Win32 automation)  
**Entry**: `python -X utf8 main.py`

---

## Project Structure

```
├── main.py                  # Entry: QApplication → App()
├── app.py                   # Main UI (App class, ~1497 lines), orchestration, poll loop
├── config.py                # Constants, SUPPORTED_LANGS, FEWSHOT_EXAMPLES, HALLUCINATIONS
├── config.json              # User secrets: hf_token, gemini_api_key, qwen_model, record_dir, inline_*
├── terms.json               # Custom glossary: {"面接": "phỏng vấn", ...}
├── requirements.txt         # Dependencies
├── AGENTS.md                # This file
├── audio/
│   ├── __init__.py          # Exports AudioLoopback, AudioMic, list_*, to_mono_16k
│   ├── loopback.py          # WASAPI loopback capture (speaker output)
│   └── mic.py               # WASAPI mic capture
├── stt/
│   ├── __init__.py          # Exports Transcriber, ReazonSpeechTranscriber, diarization symbols
│   ├── transcriber.py       # Whisper Transcriber thread (fixed + Auto VAD endpoint modes)
│   ├── sensevoice.py        # ReazonSpeech k2-asr (Japanese CPU-optimized) thread
│   └── diarization.py       # SpeakerRegistry + pyannote diarization pipeline
├── translation/
│   ├── __init__.py          # Exports QwenTranslator, TranslatorThread, terms
│   ├── qwen.py              # Qwen 3 1.7B translator + TranslatorThread (Qwen primary, Google fallback)
│   ├── inline_translate.py  # GlobalHotkey (Win32 RegisterHotKey) + InlineTranslator + ResultPopup
│   ├── terms.py             # Custom glossary load/save + _build_terms_hint()
│   └── analysis.py          # AnalysisThread: summary, keywords, issues, answer modes
├── tts/
│   ├── __init__.py          # Exports TTSThread, TTS_DEFAULT_VOICE, TTS_SPEED_OPTIONS, voices
│   └── engine.py            # TTSThread (edge-tts), pygame playback, 3-layer feedback prevention
├── ai/
│   ├── __init__.py          # Exports automation functions + AI_OPTIONS + _AI_REGISTRY
│   ├── automation.py        # Win32 window automation: Copilot, Claude Desktop, ChatGPT
│   ├── chat_dialog.py       # AI Chat popup (Gemini cloud / local Qwen, streaming)
│   └── gemini_client.py     # Cached genai.Client + fast_config + chat_config + warm_gemini
├── utils/
│   ├── __init__.py
│   ├── recorder.py          # RecordWriter: live transcript save as txt/md/srt
│   └── system_monitor.py    # SystemMonitor: psutil (RAM/CPU) + pynvml (VRAM/GPU)
├── learning/
│   ├── __init__.py
│   └── mindmap.py           # MindmapWorker + MindmapDialog: markmap HTML, AI generation, export
└── docs/
    └── img/                 # README diagrams
```

---

## Data Flow

```
Audio Source (Mic/Loopback)
  → AudioMic / AudioLoopback (PyAudio WPatch WASAPI)
    → frame_queue (raw bytes)
      → Transcriber (Whisper GPU) or ReazonSpeechTranscriber (CPU)
        → text_queue  ({"text", "speaker", "color"})
          → App.poll() → text_jp QTextEdit (left panel)
            → jp_trans_queue (seg_id, text)
              → TranslatorThread
                → QwenTranslator (primary, GPU)
                → GoogleTranslator (fallback)
                  → vi_queue (seg_id, translated_text)
                    → App.poll() → text_vi QTextEdit (right panel)
                      → tts_queue → TTSThread (edge-tts + pygame)
```

---

## Key Architectural Details

### Queues (thread-safe)
- `frame_queue` — raw audio bytes
- `text_queue` — dicts from STT `{"text", "speaker", "color"}`
- `status_queue` — status messages
- `jp_trans_queue` — `(seg_id, text)` tuples for translation
- `vi_queue` — `(seg_id, translated_text)` tuples back to UI
- `err_queue` — error strings
- `tts_queue` — text strings for TTS

### FIFO Segment Pairing
Each source segment gets `seg_id` → placeholder `⏳ ...` shown in translation panel → replaced when translation arrives.

### STT Modes
- **Fixed**: chunks at N-second intervals (1s–10s)
- **Auto (VAD endpoint)**: accumulate audio, detect silence (0.7s hang time), transcribe full sentence. Max 12s cap.

### Languages Supported
ja, en, zh, my, vi — with few-shot examples for each pair in Qwen prompt.

### 3-Layer TTS Feedback Prevention
1. `keep_recording_during_tts` Event (default ON) — don't pause STT during TTS
2. `tts_speaking` Event — transcribers skip processing while TTS plays
3. `_recent_tts` registry — dedup by normalized text match (8s TTL)

### UI App Class (`app.py:236`)
- **Poll loop** at 100ms via QTimer — drains all queues
- **Settings drawer**: QDockWidget with collapsible `_Section` widgets
- **Dark QSS theme** in `DARK_QSS` constant
- **Keyboard shortcuts**:
  - `Ctrl+Alt+T` — inline translate (replace selection globally)
  - `Ctrl+Alt+D` — inline translate (popup, non-destructive)
  - `Ctrl+Shift+C` — copy source
  - `Ctrl+Shift+V` — copy translation
  - `Ctrl+Delete` — clear all
  - `Ctrl+D` — toggle dual/single panel
  - `Ctrl+T` — toggle translation
  - `Ctrl+G` — open terms editor
  - `Ctrl+Enter` — send to selected AI
  - `Ctrl+Shift+Enter` — send to Claude

---

## Important Config Constants (`config.py`)

```python
MODEL_SIZE = env "STT_MODEL" or "medium"
DEVICE = env "STT_DEVICE" or "cuda"
COMPUTE_TYPE = env "STT_COMPUTE" or "int8_float16"
TARGET_SR = 16000
CHUNK_SEC = 4.0
OVERLAP_SEC = 0.5
RMS_THRESHOLD = 0.01
VAD_SILENCE_HANG = 0.7   # seconds of silence = sentence end
VAD_MIN_SPEECH = 0.4
VAD_MAX_SEG = 12.0
```

`save_config_value(key, value)` persists to `config.json` preserving existing keys.

---

## Config.json Format

```json
{
  "hf_token": "hf_...",
  "gemini_api_key": "...",
  "qwen_model": "Qwen/Qwen3-1.7B",
  "record_dir": "",
  "inline_enabled": false,
  "inline_from": "vi",
  "inline_to": "ja",
  "inline_engine": "Qwen local"
}
```

---

## Testing

No test framework configured. Verify by running `python -X utf8 main.py`.

---

## Common Pitfalls / Gotchas
1. **NVIDIA DLL paths** — auto-discovered in `main.py` via `importlib.util.find_spec` before any torch import
2. **UnicodeEncodeError** — always run with `-X utf8` on Windows
3. **Whisper loads before Qwen** — VRAM ordering: Whisper first, then Qwen waits via `model_ready` Event
4. **Qwen thinking** — all templates use `enable_thinking=False` to avoid `[...]` tags
5. **CUDA 12 + cuDNN 9** required for ctranslate2
6. **Diarization** requires HF_TOKEN for pyannote pipeline download
