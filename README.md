# 🎙 Interview STT — Real-time Speech-to-Text & Translation

> **Version:** `v20260622` — [Changelog](CHANGELOG.md) | [Previous version (v1.0)](../../tree/v1.0)

**Fully local, GPU-accelerated** real-time speech-to-text, translation, and AI analysis app designed for online interviews (Teams, Zoom, Google Meet). Captures speaker audio via WASAPI loopback or microphone, transcribes with Whisper or ReazonSpeech, translates with Qwen 3, and provides AI-powered content analysis — all running locally. Zero cloud cost.

![UI Mockup](docs/img/ui-mockup.svg)

---

## ✨ Features

![Features Overview](docs/img/features.svg)

### Core
- **Dual STT engine** — Whisper medium (99 languages, GPU) or ReazonSpeech k2-asr (Japanese-specialized, CPU)
- **Chunk size + Auto endpoint** — fixed 1–10s interval, or `Auto` (VAD detects the sentence boundary by silence, then translates a full sentence)
- **Local AI translation** — Qwen 3 1.7B runs entirely on GPU. Zero API cost, fully offline
- **AI Chat popup** — chat with **local Qwen** or **cloud Gemini** (switchable, streaming); insert the source/translation transcript as context
- **Speaker diarization** — pyannote.audio identifies and color-codes up to 5 speakers across chunks
- **Modern PySide6 UI** — single-row toolbar (language pill, feature toggles), settings drawer, 2 resizable panels (Source | Translation), dark theme
- **FIFO segment pairing** — Each source segment gets a `seg_id`, translation replaces the `⏳` placeholder when ready

### Productivity
- **Inline translate (global hotkey)** — select text in *any* app, press `Ctrl+Alt+T` → it's replaced in place with the translation (Qwen or Gemini)
- **Transcript recording** — save the source transcript live to a folder as `txt` / `md` / `srt`

### Audio
- **WASAPI Loopback** — Capture system/speaker audio (hear what the interviewer says)
- **Microphone input** — Capture your own voice (practice mode)
- **Device selector** — Choose specific audio device per mode

### AI Assistant Integration
- **1-click AI query** — Split button: click sends, dropdown selects AI
- **3 AI options** — Microsoft Copilot, Claude Desktop, ChatGPT
- **Window automation** — Auto-find window → focus → click input → paste → send (Win32 API)

### TTS (Text-to-Speech)
- **Edge TTS** — Microsoft Neural Voices for reading translations aloud
- **TTS feedback prevention (3-layer)** — keeps recording during TTS (no data loss) while rejecting its own voice via language filter + recent-TTS-text dedup

### Quality
- **VAD filter** — Voice Activity Detection skips silence
- **Hallucination filter** — Per-language blacklist removes common Whisper artifacts
- **RMS threshold** — Ignores audio below volume threshold
- **Custom terms** — Edit interview vocabulary (JP↔VI) via built-in editor, hot-reload into Qwen prompt

---

## 🏗 Architecture

![Architecture](docs/img/architecture.svg)

---

## 🔄 Data Flow

![Data Flow](docs/img/flow.svg)

---

## 📋 Requirements

| Component | Spec |
|---|---|
| **OS** | Windows 10/11 |
| **GPU** | NVIDIA with ≥6 GB VRAM (RTX 3060+ recommended) |
| **CUDA** | 12.x with cuDNN 9 |
| **Python** | 3.10+ |
| **RAM** | 8 GB+ |

### VRAM / Resource Usage
| Model | VRAM | Device |
|---|---|---|
| Whisper medium | ~2 GB | GPU |
| Qwen 3 1.7B | ~2.5 GB | GPU |
| ReazonSpeech k2 | ~159 MB | CPU |
| **Total (Whisper)** | **~4.5 GB** | |
| **Total (ReazonSpeech)** | **~2.5 GB** | |

---

## 🚀 Installation

### 1. Clone

```bash
git clone https://github.com/KhanhNguyenAI/translate-For-E-learning-using-AI-LOCAL.git
cd translate-For-E-learning-using-AI-LOCAL
```

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

Copy the example config and add your tokens:

```bash
cp config.example.json config.json
```

Edit `config.json`:

```json
{
  "hf_token": "hf_YOUR_HUGGINGFACE_TOKEN",
  "gemini_api_key": "YOUR_GEMINI_API_KEY",
  "qwen_model": "Qwen/Qwen3-1.7B",
  "record_dir": "",
  "inline_enabled": false,
  "inline_from": "vi",
  "inline_to": "ja",
  "inline_engine": "Qwen local"
}
```

| Key | Required | Purpose |
|---|---|---|
| `hf_token` | For diarization | HuggingFace token for pyannote.audio |
| `gemini_api_key` | Optional | Gemini AI Chat + inline translate (cloud engine) |
| `qwen_model` | Optional | Override Qwen model name (default: Qwen/Qwen3-1.7B) |
| `record_dir` | Optional | Default folder for transcript recording |
| `inline_*` | Optional | Inline-translate settings (set from the UI) |

### 5. Run

```bash
python -X utf8 main.py
```

> **Note:** `-X utf8` is required on Windows to avoid UnicodeEncodeError with Vietnamese output.

---

## 🎮 Usage

### Basic Workflow

1. **Select languages** — Choose input language (e.g. 🇯🇵 Japanese) and output language (e.g. 🇻🇳 Vietnamese)
2. **Select STT engine** — Whisper (multi-language, GPU) or ReazonSpeech (Japanese, CPU)
3. **Select chunk size** — 1s (fastest) to 10s (most context) transcription interval
4. **Select audio source** — `🔊 Loa` (speaker/loopback) for interviews, `🎙 Mic` for practice
5. **Click ▶ Start** — STT engine loads first, then Qwen loads (VRAM ordering)
6. **Toggle features** — 🌐 Translate, 🔊 TTS, 👥 Speakers (diarization) — all OFF by default
7. **Watch real-time transcription** — Left panel shows source text, right shows translation
8. **AI Chat** — Click 🧠 AI Chat to open the popup; chat with local Qwen or cloud Gemini, insert the transcript as context
9. **Ask AI** — Click the AI button to send transcribed text to Copilot/Claude/ChatGPT
10. **Settings drawer (⚙)** — engine, chunk/Auto, devices, TTS, recording folder/format, inline translate

### Supported Languages

| Language | Code | Flag |
|---|---|---|
| Japanese | `ja` | 🇯🇵 |
| English | `en` | 🇬🇧 |
| Chinese | `zh` | 🇨🇳 |
| Myanmar | `my` | 🇲🇲 |
| Vietnamese | `vi` | 🇻🇳 |

Whisper supports 99 languages total — add more in `SUPPORTED_LANGS` dict.

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Alt+T` | **Inline translate** — replace selected text in any app (global) |
| `Ctrl+Enter` | Send to selected AI (Copilot) |
| `Ctrl+Shift+Enter` | Send to Claude |
| `Ctrl+Shift+C` | Copy source text |
| `Ctrl+Shift+V` | Copy translation |
| `Ctrl+Delete` | Clear all text |
| `Ctrl+D` | Toggle dual/single panel |
| `Ctrl+T` | Toggle translation on/off |
| `Ctrl+G` | Open terms editor |

### Custom Terms

Click `⚙` to open the terms editor. Format: one term per line, `Japanese = Vietnamese`:

```
面接 = phỏng vấn
志望動機 = động cơ ứng tuyển
自己紹介 = tự giới thiệu
```

Terms are saved to `terms.json` and hot-reloaded into the Qwen translation prompt.

---

## 📁 Project Structure

```
├── main.py                  # Entry point (QApplication)
├── app.py                   # Main PySide6 UI + orchestration
├── config.py                # Settings, constants, model config
├── requirements.txt         # Python dependencies
├── CHANGELOG.md             # Version history
├── audio/
│   ├── loopback.py          # WASAPI loopback capture
│   └── mic.py               # Microphone capture
├── stt/
│   ├── transcriber.py       # Whisper STT (fixed + Auto VAD endpoint)
│   ├── sensevoice.py        # ReazonSpeech STT (fixed + Auto VAD endpoint)
│   └── diarization.py       # Speaker diarization (pyannote)
├── translation/
│   ├── qwen.py              # Qwen 3 translator thread
│   ├── inline_translate.py  # Global-hotkey inline translate (Ctrl+Alt+T / D)
│   └── terms.py             # Custom terminology management
├── tts/
│   └── engine.py            # Edge TTS + 3-layer feedback prevention
├── ai/
│   ├── automation.py        # Copilot/Claude/ChatGPT window automation
│   ├── gemini_client.py     # Cached Gemini client + fast config
│   └── chat_dialog.py       # AI Chat popup (Gemini / local Qwen)
├── learning/
│   └── mindmap.py           # Memory mind map (markmap) + editor
├── utils/
│   ├── recorder.py          # Transcript recording (txt/md/srt)
│   └── system_monitor.py    # RAM / VRAM / CPU / GPU monitor
└── docs/
    └── img/                 # README diagrams + screenshots
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `medium` | Whisper model size (tiny/base/small/medium/large) |
| `STT_DEVICE` | `cuda` | Device for Whisper (cuda/cpu) |
| `STT_COMPUTE` | `int8_float16` | Compute type for CTranslate2 |

Example:

```bash
set STT_MODEL=large
python -X utf8 main.py
```

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `torch` is CPU-only | Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Whisper model is None | VRAM issue — Qwen waits for Whisper via `model_ready` Event |
| UnicodeEncodeError | Run with `python -X utf8 main.py` |
| No loopback devices | Install [PyAudioWPatch](https://pypi.org/project/PyAudioWPatch/) and check WASAPI |
| Qwen returns Japanese | Few-shot examples + `enable_thinking=False` should fix this |
| NVIDIA DLL not found | The app auto-discovers DLL paths via `importlib.util.find_spec` |

---

## 📄 License

MIT License — free for personal and commercial use.

---

## 🙏 Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 Whisper implementation
- [Qwen 3](https://huggingface.co/Qwen/Qwen3-1.7B) — Alibaba's multilingual LLM
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) — WASAPI loopback support
- [ReazonSpeech](https://research.reazon.jp/projects/ReazonSpeech/) — Japanese-specialized STT
- [Edge TTS](https://github.com/rany2/edge-tts) — Microsoft Neural Voices

---

**Made with ❤️ for the Vietnamese community studying and working in Japan**
