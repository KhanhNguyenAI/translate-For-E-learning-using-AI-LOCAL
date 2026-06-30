# -*- coding: utf-8 -*-
"""
Configuration constants, language settings, and config.json loading.
"""

import os
import json

# ----------------------------- CẤU HÌNH -----------------------------
MODEL_SIZE   = os.environ.get("STT_MODEL", "medium")
TARGET_SR    = 16000
CHUNK_SEC    = 4.0
OVERLAP_SEC  = 0.5
DEVICE       = os.environ.get("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE", "int8_float16")

# Ngưỡng âm lượng tối thiểu — bỏ qua nếu quá im (tránh hallucination)
RMS_THRESHOLD = 0.01

# ── Auto (endpoint) mode — VAD-based sentence segmentation ─────────────
VAD_SILENCE_HANG = 0.7   # giây im lặng liên tục để coi là hết câu
VAD_MIN_SPEECH   = 0.4   # độ dài tối thiểu (giây) mới đáng nhận dạng
VAD_MAX_SEG      = 12.0  # trần tối đa (giây) — nói dài không nghỉ thì tự cắt

DEFAULT_FONT_SIZE = 15

# ── config.json loading ─────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_config_data = {}
try:
    _config_data = json.load(open(_CONFIG_PATH))
except Exception:
    pass

HF_TOKEN = _config_data.get("hf_token")
QWEN_MODEL_NAME = _config_data.get("qwen_model", "Qwen/Qwen3-1.7B")
GEMINI_API_KEY = _config_data.get("gemini_api_key")
GEMINI_MODEL = _config_data.get("gemini_model", "gemini-2.5-flash")
RECORD_DIR = _config_data.get("record_dir")

# ── Inline translate (global hotkey) ──────────────────────────────────
INLINE_FROM    = _config_data.get("inline_from", "vi")
INLINE_TO      = _config_data.get("inline_to", "ja")
INLINE_ENGINE  = _config_data.get("inline_engine", "Qwen local")
INLINE_ENABLED = _config_data.get("inline_enabled", False)


def save_config_value(key, value):
    """Persist a single key back into config.json (preserves existing keys)."""
    _config_data[key] = value
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_config_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[config] save failed: {e}")

# ── Multi-language config ─────────────────────────────────────────────
SUPPORTED_LANGS = {
    "ja": {"name": "Japanese",  "flag": "\U0001f1ef\U0001f1f5", "whisper": "ja"},
    "en": {"name": "English",   "flag": "\U0001f1ec\U0001f1e7", "whisper": "en"},
    "zh": {"name": "Chinese",   "flag": "\U0001f1e8\U0001f1f3", "whisper": "zh"},
    "my": {"name": "Myanmar",   "flag": "\U0001f1f2\U0001f1f2", "whisper": "my"},
    "vi": {"name": "Vietnamese","flag": "\U0001f1fb\U0001f1f3", "whisper": "vi"},
}

LANG_NAMES_EN = {
    "ja": "Japanese", "en": "English", "zh": "Chinese",
    "my": "Myanmar (Burmese)", "vi": "Vietnamese",
}

# Few-shot examples per (source, target) pair for Qwen
FEWSHOT_EXAMPLES = {
    ("ja", "vi"): [
        ("自己紹介をお願いします", "Xin hãy tự giới thiệu bản thân"),
        ("どうしてこの会社を選びましたか", "Tại sao bạn chọn công ty này?"),
    ],
    ("en", "vi"): [
        ("Please introduce yourself", "Xin hãy tự giới thiệu bản thân"),
        ("What are your strengths?", "Điểm mạnh của bạn là gì?"),
    ],
    ("zh", "vi"): [
        ("请自我介绍一下", "Xin hãy tự giới thiệu bản thân"),
        ("你为什么选择我们公司？", "Tại sao bạn chọn công ty chúng tôi?"),
    ],
    ("my", "vi"): [
        ("ကိုပ္အကြောင်း မိတ္ဆက်ပေးပါ", "Xin hãy tự giới thiệu bản thân"),
        ("ဘာကြောင့် ဒီကုမ္ပဏီကို ရွေးချယ်တာလဲ", "Tại sao bạn chọn công ty này?"),
    ],
    ("ja", "en"): [
        ("自己紹介をお願いします", "Please introduce yourself"),
        ("どうしてこの会社を選びましたか", "Why did you choose this company?"),
    ],
    ("ja", "zh"): [
        ("自己紹介をお願いします", "请做一下自我介绍"),
        ("どうしてこの会社を選びましたか", "你为什么选择了这家公司？"),
    ],
    ("en", "ja"): [
        ("Please introduce yourself", "自己紹介をお願いします"),
        ("What are your strengths?", "あなたの強みは何ですか？"),
    ],
    ("zh", "ja"): [
        ("请自我介绍一下", "自己紹介をお願いします"),
        ("你为什么选择我们公司？", "どうして弊社を選びましたか？"),
    ],
}

# Hallucination blacklist per language
HALLUCINATIONS = {
    "ja": {
        "ご視聴ありがとうございました", "ありがとうございました",
        "チャンネル登録よろしくお願いします", "お疲れ様でした",
        "ご清聴ありがとうございました", "字幕は自動生成されています",
        "お願いします", "よろしくお願いいたします",
    },
    "en": {
        "Thank you for watching.", "Please subscribe.",
        "Thanks for watching.", "Subtitles by the Amara.org community",
    },
    "zh": {
        "感谢收看", "请订阅", "字幕由Amara.org社区提供",
    },
}
