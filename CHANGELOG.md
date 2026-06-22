# Changelog

## v20260622 (2026-06-22)

### New Features
- **ReazonSpeech STT** — Japanese-specialized engine (k2-asr, 159M params, CPU-based) as alternative to Whisper
- **STT Engine selector** — Dropdown to switch between Whisper and ReazonSpeech
- **Chunk size selector** — Adjustable transcription interval (1s / 2s / 4s / 6s / 8s / 10s)
- **AI Analysis panel** — Qwen-powered analysis with 4 modes:
  - 📝 Tóm tắt (Summary)
  - 🔑 Keywords
  - ⚠️ Vấn đề (Issues)
  - 💡 Gợi ý trả lời (Answer suggestions)
- **TTS feedback prevention** — `tts_speaking` flag pauses transcription during TTS playback
- **Analysis toggle** — 🧠 checkbox to show/hide AI Analysis panel

### Changes
- Diarization, Translation, TTS default to OFF
- AI Analysis outputs in STT source language
- Copilot/ChatGPT click positions adjusted

---

## v1.0

### Initial Release
- Real-time STT with Whisper (medium, GPU)
- Local AI translation with Qwen 3 1.7B
- Speaker diarization (pyannote.audio)
- Dual panel UI (Source + Translation)
- WASAPI Loopback + Microphone input
- AI Assistant integration (Copilot, Claude, ChatGPT)
- Custom interview terms editor
- TTS with Edge TTS
