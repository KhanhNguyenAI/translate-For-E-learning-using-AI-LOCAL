# -*- coding: utf-8 -*-
"""
Interview STT - Realtime Japanese speech-to-text from speaker (WASAPI loopback)
Chạy: python main.py
"""

import os
import sys
import time
import queue
import threading
import json

# Fix Windows: tìm đúng vị trí NVIDIA DLLs bằng importlib
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

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("Thiếu PyAudioWPatch. Chạy: pip install PyAudioWPatch")
    sys.exit(1)

from faster_whisper import WhisperModel

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False
    print("[WARN] deep-translator chưa cài — fallback Google Translate tắt")

# ========================= QWEN TRANSLATOR ============================
HAS_QWEN = False
_qwen_model = None
_qwen_tokenizer = None

QWEN_MODEL_NAME = "Qwen/Qwen3-1.7B"
try:
    _cfg_q = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
    QWEN_MODEL_NAME = _cfg_q.get("qwen_model", QWEN_MODEL_NAME)
except Exception:
    pass

TERMS_PATH = os.path.join(os.path.dirname(__file__), "terms.json")
_custom_terms = {}
try:
    _custom_terms = json.load(open(TERMS_PATH, encoding="utf-8"))
except Exception:
    pass


def _build_terms_hint() -> str:
    if not _custom_terms:
        return ""
    pairs = [f"{k} = {v}" for k, v in _custom_terms.items()]
    return "Bảng thuật ngữ:\n" + "\n".join(pairs) + "\n\n"


class QwenTranslator:
    """Dịch đa ngôn ngữ bằng Qwen 3 1.7B chạy local GPU."""

    def __init__(self, model_name: str = None, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        name = model_name or QWEN_MODEL_NAME
        print(f"[Qwen] Loading {name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._device = self.model.device
        self._terms_hint = _build_terms_hint()
        print(f"[Qwen] Ready on {self._device}")

    def translate(self, text: str, src_lang: str = "ja", tgt_lang: str = "vi") -> str:
        import torch, re
        if not text.strip():
            return ""

        src_name = LANG_NAMES_EN.get(src_lang, src_lang)
        tgt_name = LANG_NAMES_EN.get(tgt_lang, tgt_lang)

        system_msg = (
            f"You are a {src_name}-{tgt_name} translator. "
            f"Translate the {src_name} text to {tgt_name}. "
            f"Reply with ONLY the {tgt_name} translation. "
            f"Do NOT repeat the source text. Do NOT explain.\n"
            + self._terms_hint
        )
        messages = [{"role": "system", "content": system_msg}]

        examples = FEWSHOT_EXAMPLES.get((src_lang, tgt_lang), [])
        for src_ex, tgt_ex in examples:
            messages.append({"role": "user", "content": f"Translate to {tgt_name}: {src_ex}"})
            messages.append({"role": "assistant", "content": tgt_ex})

        messages.append({"role": "user", "content": f"Translate to {tgt_name}: {text}"})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result


def load_qwen_translator() -> QwenTranslator | None:
    global HAS_QWEN
    try:
        translator = QwenTranslator()
        HAS_QWEN = True
        return translator
    except Exception as e:
        print(f"[Qwen] Load failed: {e}")
        HAS_QWEN = False
        return None
# ====================================================================

# ========================= DIARIZATION ==============================

HF_TOKEN = None
try:
    _cfg = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
    HF_TOKEN = _cfg.get("hf_token")
except Exception:
    pass

_diarize_pipeline = None
HAS_DIARIZATION = False

def load_diarization_pipeline():
    global _diarize_pipeline, HAS_DIARIZATION
    try:
        import warnings; warnings.filterwarnings("ignore")
        import torch
        from pyannote.audio import Pipeline
        _diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        )
        HAS_DIARIZATION = True
        print("[Diarization] Pipeline loaded OK")
    except Exception as e:
        print(f"[Diarization] Load failed: {e}")
        HAS_DIARIZATION = False

# Màu theo speaker index
SPEAKER_COLORS = ["#64b5f6", "#81c784", "#ffb74d", "#f06292", "#ba68c8"]
SPEAKER_LABELS = ["🎙 Speaker A", "🎙 Speaker B", "🎙 Speaker C", "🎙 Speaker D", "🎙 Speaker E"]


class SpeakerRegistry:
    """Theo dõi speaker identity xuyên suốt các chunk bằng embedding similarity."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.profiles  = {}   # global_label -> embedding tensor
        self.next_idx  = 0

    def resolve(self, local_id: str, embeddings: dict) -> str:
        """Map local speaker id (SPEAKER_00...) → global label nhất quán."""
        import torch, torch.nn.functional as F
        emb = embeddings.get(local_id)
        if emb is None:
            return self._new_label()
        emb = emb.detach().cpu().flatten().unsqueeze(0)

        best_label, best_sim = None, -1.0
        for label, prof in self.profiles.items():
            sim = F.cosine_similarity(emb, prof.flatten().unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_label = sim, label

        if best_sim >= self.threshold and best_label:
            # Cùng người — cập nhật running average
            self.profiles[best_label] = (0.85 * self.profiles[best_label]
                                         + 0.15 * emb.squeeze(0))
            return best_label
        else:
            # Người mới
            label = self._new_label()
            self.profiles[label] = emb.squeeze(0)
            return label

    def _new_label(self) -> str:
        label = f"SPEAKER_{self.next_idx:02d}"
        self.next_idx += 1
        return label

    def reset(self):
        self.profiles.clear()
        self.next_idx = 0


_speaker_registry = SpeakerRegistry()


def diarize_audio(audio_np: np.ndarray) -> list[tuple[float, float, str]]:
    """Trả về list (start, end, global_speaker_label) — nhất quán xuyên chunk."""
    if not HAS_DIARIZATION or _diarize_pipeline is None:
        return []
    try:
        import torch
        tensor = torch.tensor(audio_np).unsqueeze(0).float()
        out    = _diarize_pipeline({"waveform": tensor, "sample_rate": TARGET_SR})
        ann    = out.speaker_diarization
        labels = ann.labels()   # ['SPEAKER_00', 'SPEAKER_01', ...]

        # speaker_embeddings: numpy array (N, 256) — rows theo thứ tự labels
        embeds = {}
        raw_emb = getattr(out, "speaker_embeddings", None)
        if raw_emb is not None and len(raw_emb) == len(labels):
            import torch
            for i, lbl in enumerate(labels):
                embeds[lbl] = torch.tensor(raw_emb[i])

        results = []
        for turn, _, local_spk in ann.itertracks(yield_label=True):
            global_spk = _speaker_registry.resolve(local_spk, embeds)
            results.append((turn.start, turn.end, global_spk))
        return results
    except Exception as e:
        print(f"[Diarization] Error: {e}")
        return []


def assign_speaker_color(global_speaker_id: str) -> tuple[str, str]:
    """Gán màu và label từ global speaker id (SPEAKER_00, SPEAKER_01...)."""
    try:
        idx = int(global_speaker_id.split("_")[-1]) % len(SPEAKER_COLORS)
    except Exception:
        idx = 0
    return SPEAKER_COLORS[idx], SPEAKER_LABELS[idx]
# ====================================================================

# -------- Mở / Focus Microsoft Copilot (dùng ctypes built-in) -------
import ctypes, ctypes.wintypes, subprocess

# Microsoft Copilot app (Windows Store) — thử cả 2 package ID phổ biến
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


# -------- Mở / Focus Claude Desktop (dùng ctypes built-in) ----------
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


# -------- Mở / Focus ChatGPT (dùng ctypes built-in) -----------------
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
# ---------------------------------------------------------------------

# ----------------------------- CẤU HÌNH -----------------------------
MODEL_SIZE   = os.environ.get("STT_MODEL", "medium")
TARGET_SR    = 16000
CHUNK_SEC    = 4.0
OVERLAP_SEC  = 0.5
DEVICE       = os.environ.get("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE", "int8_float16")

# Ngưỡng âm lượng tối thiểu — bỏ qua nếu quá im (tránh hallucination)
RMS_THRESHOLD = 0.01

# ── Multi-language config ─────────────────────────────────────────────
SUPPORTED_LANGS = {
    "ja": {"name": "Tiếng Nhật",  "flag": "🇯🇵", "whisper": "ja"},
    "en": {"name": "Tiếng Anh",   "flag": "🇬🇧", "whisper": "en"},
    "zh": {"name": "Tiếng Trung",  "flag": "🇨🇳", "whisper": "zh"},
    "my": {"name": "Myanmar",      "flag": "🇲🇲", "whisper": "my"},
    "vi": {"name": "Tiếng Việt",   "flag": "🇻🇳", "whisper": "vi"},
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
        ("ကိုယ့်အကြောင်း မိတ်ဆက်ပေးပါ", "Xin hãy tự giới thiệu bản thân"),
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
# --------------------------------------------------------------------


def list_loopback_devices():
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for dev in pa.get_loopback_device_info_generator():
            devices.append((int(dev["index"]), dev["name"]))
    finally:
        pa.terminate()
    return devices


def list_mic_devices():
    pa = pyaudio.PyAudio()
    devices = []
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (info.get("hostApi") == wasapi["index"]
                    and int(info.get("maxInputChannels", 0)) > 0
                    and "loopback" not in info.get("name", "").lower()):
                devices.append((int(info["index"]), info["name"]))
    finally:
        pa.terminate()
    return devices


# ========================= AUDIO LOOPBACK ===========================
class AudioLoopback:
    def __init__(self, frame_queue, device_index=None):
        self.pa = pyaudio.PyAudio()
        self.frame_queue = frame_queue
        self.device_index = device_index
        self.stream = None
        self.device_rate = TARGET_SR
        self.channels = 1
        self._running = False

    def _find_device(self):
        if self.device_index is not None:
            return self.pa.get_device_info_by_index(self.device_index)
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for dev in self.pa.get_loopback_device_info_generator():
            if default_out["name"] in dev["name"]:
                return dev
        for dev in self.pa.get_loopback_device_info_generator():
            return dev
        raise RuntimeError("Không tìm thấy WASAPI loopback nào.")

    def start(self):
        dev = self._find_device()
        self.device_rate = int(dev["defaultSampleRate"])
        self.channels = max(1, int(dev["maxInputChannels"]))
        print(f"[Audio] {dev['name']} | {self.device_rate}Hz | {self.channels}ch")
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.device_rate,
            frames_per_buffer=1024,
            input=True,
            input_device_index=int(dev["index"]),
            stream_callback=self._callback,
        )
        self._running = True
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        if self._running:
            self.frame_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def stop(self):
        self._running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def terminate(self):
        self.stop()
        self.pa.terminate()


class AudioMic:
    """Thu âm từ microphone (WASAPI input, không phải loopback)."""

    def __init__(self, frame_queue, device_index=None):
        self.pa = pyaudio.PyAudio()
        self.frame_queue = frame_queue
        self.device_index = device_index
        self.stream = None
        self.device_rate = TARGET_SR
        self.channels = 1
        self._running = False

    def _find_device(self):
        if self.device_index is not None:
            return self.pa.get_device_info_by_index(self.device_index)
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_in = self.pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        return default_in

    def start(self):
        dev = self._find_device()
        self.device_rate = int(dev["defaultSampleRate"])
        self.channels = max(1, int(dev["maxInputChannels"]))
        print(f"[Mic] {dev['name']} | {self.device_rate}Hz | {self.channels}ch")
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.device_rate,
            frames_per_buffer=1024,
            input=True,
            input_device_index=int(dev["index"]),
            stream_callback=self._callback,
        )
        self._running = True
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        if self._running:
            self.frame_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def stop(self):
        self._running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def terminate(self):
        self.stop()
        self.pa.terminate()


def to_mono_16k(raw_bytes, device_rate, channels):
    audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if device_rate != TARGET_SR and len(audio) > 0:
        n_out = int(round(len(audio) * TARGET_SR / device_rate))
        if n_out > 0:
            x_old = np.linspace(0, 1, len(audio), endpoint=False)
            x_new = np.linspace(0, 1, n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


# ====================================================================

# ========================= TRANSCRIBER ==============================
class Transcriber(threading.Thread):
    def __init__(self, frame_queue, text_queue, status_queue, get_params,
                 use_diarization=None, get_src_lang=None):
        super().__init__(daemon=True)
        self.frame_queue      = frame_queue
        self.text_queue       = text_queue
        self.status_queue     = status_queue
        self.get_params       = get_params
        self._use_diarization = use_diarization  # tk.BooleanVar
        self._get_src_lang    = get_src_lang or (lambda: "ja")
        self.model = None
        self._stop = threading.Event()
        self.model_ready = threading.Event()

    def load_model(self):
        self.status_queue.put(f"Đang tải model '{MODEL_SIZE}' ({DEVICE})...")
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            print(f"[Whisper] GPU error: {e}")
            self.status_queue.put(f"GPU lỗi ({e}); chuyển CPU...")
            try:
                self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            except Exception as e2:
                print(f"[Whisper] CPU error: {e2}")
                self.status_queue.put(f"❌ Whisper load thất bại: {e2}")
                return
        self.model_ready.set()  # báo hiệu Whisper đã load xong
        if HAS_DIARIZATION:
            self.status_queue.put("Sẵn sàng (Diarization ON). Đang nghe loa...")
        else:
            self.status_queue.put("Sẵn sàng. Đang nghe loa...")

    def run(self):
        self.load_model()
        buf = np.zeros(0, dtype=np.float32)
        overlap = np.zeros(0, dtype=np.float32)
        chunk_samples = int(CHUNK_SEC * TARGET_SR)
        last_vol_t = time.time()

        while not self._stop.is_set():
            try:
                raw = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            device_rate, channels = self.get_params()
            mono = to_mono_16k(raw, device_rate, channels)
            buf = np.concatenate([buf, mono])

            now = time.time()
            if now - last_vol_t > 1.5 and len(buf) > 0:
                rms = float(np.sqrt(np.mean(buf**2)))
                filled = int(min(rms * 300, 20))
                bar = "█" * filled + "░" * (20 - filled)
                self.status_queue.put(f"Đang nghe...  [{bar}]  {rms:.4f}")
                last_vol_t = now

            if len(buf) >= chunk_samples:
                audio_in = np.concatenate([overlap, buf])
                overlap = buf[-int(OVERLAP_SEC * TARGET_SR):].copy()
                buf = np.zeros(0, dtype=np.float32)
                self._transcribe(audio_in)

    def _transcribe(self, audio):
        try:
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < RMS_THRESHOLD:
                return
            if self.model is None:
                return
            diar_on = (self._use_diarization is None or self._use_diarization.get())
            diar_future = []
            if HAS_DIARIZATION and diar_on:
                t = threading.Thread(
                    target=lambda: diar_future.append(diarize_audio(audio)),
                    daemon=True,
                )
                t.start()

            cur_lang = self._get_src_lang()
            segments, _ = self.model.transcribe(
                audio, language=cur_lang, beam_size=2,
                vad_filter=True, condition_on_previous_text=False,
                no_speech_threshold=0.5,
                word_timestamps=True,
            )
            seg_list = list(segments)

            all_words = []
            for seg in seg_list:
                for w in (seg.words or []):
                    all_words.append(w)

            full_text = "".join(s.text for s in seg_list).strip()
            if not full_text:
                return
            stripped = full_text.replace("。","").replace("、","").replace(" ","").strip()
            hall_set = HALLUCINATIONS.get(cur_lang, set())
            if stripped in hall_set:
                return

            if HAS_DIARIZATION and diar_on:
                t.join(timeout=8.0)
                turns = diar_future[0] if diar_future else []

                if turns and all_words:
                    # Khớp từng word → diarization turn → gom theo speaker liên tiếp
                    def find_speaker(t_mid):
                        for start, end, spk in turns:
                            if start <= t_mid <= end:
                                return spk
                        # Nếu không khớp → lấy turn gần nhất
                        return min(turns, key=lambda x: abs((x[0]+x[1])/2 - t_mid))[2]

                    # Gom words liên tiếp cùng speaker thành 1 segment
                    grouped = []
                    for w in all_words:
                        t_mid = (w.start + w.end) / 2
                        spk   = find_speaker(t_mid)
                        if grouped and grouped[-1][0] == spk:
                            grouped[-1][1].append(w.word)
                        else:
                            grouped.append([spk, [w.word]])

                    for spk, words_list in grouped:
                        seg_text = "".join(words_list).strip()
                        if not seg_text:
                            continue
                        color, label = assign_speaker_color(spk)
                        self.text_queue.put({"text": seg_text, "speaker": label, "color": color})
                    return

                elif turns:
                    # Không có word timestamps → fallback dominant speaker
                    from collections import Counter
                    spk_times = Counter()
                    for s, e, spk in turns:
                        spk_times[spk] += (e - s)
                    dominant = spk_times.most_common(1)[0][0]
                    color, label = assign_speaker_color(dominant)
                    self.text_queue.put({"text": full_text, "speaker": label, "color": color})
                    return

            self.text_queue.put({"text": full_text, "speaker": None, "color": "#eee"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_queue.put(f"Lỗi nhận dạng: {e}")

    def stop(self):
        self._stop.set()


# ========================= TRANSLATOR THREAD ========================
class TranslatorThread(threading.Thread):
    """Nhận text từ queue, dịch bằng Qwen (ưu tiên) hoặc Google Translate."""

    def __init__(self, src_queue, tgt_queue, err_queue, status_queue=None,
                 wait_for_event: threading.Event = None,
                 get_lang_pair=None):
        super().__init__(daemon=True)
        self.src_queue      = src_queue
        self.tgt_queue      = tgt_queue
        self.err_queue      = err_queue
        self.status_queue   = status_queue
        self._wait_for      = wait_for_event
        self._get_lang_pair = get_lang_pair or (lambda: ("ja", "vi"))
        self._stop = threading.Event()
        self._qwen = None
        self._google = None
        self._google_pair = (None, None)
        self.engine_name = "none"

    def _set_status(self, msg):
        if self.status_queue:
            self.status_queue.put(msg)

    def run(self):
        if self._wait_for:
            self._set_status("⏳ Đợi Whisper load xong...")
            self._wait_for.wait(timeout=120)

        self._set_status("🤖 Đang tải Qwen 3 dịch thuật...")
        self._qwen = load_qwen_translator()
        if self._qwen:
            self.engine_name = "Qwen 3 1.7B"
            self._set_status("🤖 Dịch thuật: Qwen 3 1.7B (local GPU)")
        else:
            if HAS_TRANSLATOR:
                src, tgt = self._get_lang_pair()
                try:
                    self._google = GoogleTranslator(source=src, target=tgt)
                    self._google_pair = (src, tgt)
                    self.engine_name = "Google Translate"
                    self._set_status("🌐 Dịch thuật: Google Translate (fallback)")
                except Exception as e:
                    self.err_queue.put(f"Không khởi tạo được translator: {e}")
                    return
            else:
                self.err_queue.put("Không có engine dịch nào khả dụng")
                return

        while not self._stop.is_set():
            try:
                item = self.src_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if isinstance(item, tuple):
                seg_id, src_text = item
            else:
                seg_id, src_text = None, item

            src_lang, tgt_lang = self._get_lang_pair()

            try:
                if self._qwen:
                    tgt_text = self._qwen.translate(src_text, src_lang, tgt_lang)
                elif self._google:
                    if (src_lang, tgt_lang) != self._google_pair:
                        self._google = GoogleTranslator(source=src_lang, target=tgt_lang)
                        self._google_pair = (src_lang, tgt_lang)
                    tgt_text = self._google.translate(src_text)
                else:
                    continue
                if tgt_text:
                    self.tgt_queue.put((seg_id, tgt_text))
            except Exception as e:
                self.err_queue.put(f"Lỗi dịch ({self.engine_name}): {e}")

    def stop(self):
        self._stop.set()


# ========================= APP CHÍNH ================================
import tkinter as tk
from tkinter import ttk, scrolledtext

DEFAULT_FONT_SIZE = 15


class App:
    def __init__(self, root):
        self.root = root
        root.title("Interview STT — Multi-Language")
        root.attributes("-topmost", True)
        root.geometry("960x520+40+40")

        self.frame_queue    = queue.Queue()
        self.text_queue     = queue.Queue()
        self.status_queue   = queue.Queue()
        self.jp_trans_queue = queue.Queue()   # src text queue (kept name for compat)
        self.vi_queue       = queue.Queue()   # tgt text queue
        self.err_queue      = queue.Queue()

        self.audio = None
        self.transcriber = None
        self.translator_thread = None
        self.running = False
        self._last_speaker = None
        self._font_size = DEFAULT_FONT_SIZE
        self._dual_mode = True
        self._translate_on = True
        self._seg_counter = 0
        self._pending_vi = {}
        self._vi_next_id = 0

        # ── Language dropdown options ─────────────────────────────────
        self._lang_codes = list(SUPPORTED_LANGS.keys())
        lang_display = [f"{SUPPORTED_LANGS[c]['flag']} {SUPPORTED_LANGS[c]['name']}"
                        for c in self._lang_codes]

        # ── Toolbar ────────────────────────────────────────────────────
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=(6, 2))

        # ─ Group 1: Language pair ─
        self._src_lang_var = tk.StringVar()
        self._src_combo = ttk.Combobox(
            top, textvariable=self._src_lang_var,
            values=lang_display, state="readonly", width=13,
        )
        self._src_combo.current(0)  # ja
        self._src_combo.pack(side="left", padx=(0, 2))
        self._src_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        tk.Label(top, text="→", font=("Segoe UI", 10, "bold"),
                 fg="#888").pack(side="left")

        self._tgt_lang_var = tk.StringVar()
        self._tgt_combo = ttk.Combobox(
            top, textvariable=self._tgt_lang_var,
            values=lang_display, state="readonly", width=13,
        )
        self._tgt_combo.current(self._lang_codes.index("vi"))
        self._tgt_combo.pack(side="left", padx=(2, 4))
        self._tgt_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        # ─ Start button ─
        self.btn = tk.Button(top, text="▶ Start", width=8, command=self.toggle,
                             bg="#2e7d32", fg="white", font=("Segoe UI", 10, "bold"))
        self.btn.pack(side="left", padx=(0, 2))

        # ─ Separator ─
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        # ─ Group 2: Text actions ─
        tk.Button(top, text="🗑", width=3,
                  command=self.clear).pack(side="left", padx=1)
        tk.Button(top, text="📋L", width=3,
                  command=self.copy_jp).pack(side="left", padx=1)
        tk.Button(top, text="📋R", width=3,
                  command=self.copy_vi).pack(side="left", padx=1)

        # ─ Separator ─
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        # ─ Group 3: AI — click gửi, ▾ chọn model ─
        AI_OPTIONS = {
            "Copilot":  {"color": "#0078d4", "icon": "🟦"},
            "Claude":   {"color": "#d97706", "icon": "🟧"},
            "ChatGPT":  {"color": "#10a37f", "icon": "🟩"},
        }
        self._ai_choice = "Copilot"

        ai_frame = tk.Frame(top)
        ai_frame.pack(side="left", padx=2)

        self._ai_send_btn = tk.Button(
            ai_frame, text=f"🤖 Copilot", width=10,
            command=self._send_to_ai,
            bg="#0078d4", fg="white", font=("Segoe UI", 9, "bold"),
        )
        self._ai_send_btn.pack(side="left")

        self._ai_drop = tk.Menubutton(
            ai_frame, text="▾", width=2,
            bg="#0078d4", fg="white", font=("Segoe UI", 9, "bold"),
            relief="raised", activebackground="#005a9e", activeforeground="white",
        )
        self._ai_menu = tk.Menu(self._ai_drop, tearoff=0,
                                 bg="#21262d", fg="#c9d1d9",
                                 activebackground="#30363d", activeforeground="white",
                                 font=("Segoe UI", 10))
        for ai_name, ai_cfg in AI_OPTIONS.items():
            self._ai_menu.add_command(
                label=f"{ai_cfg['icon']}  {ai_name}",
                command=lambda n=ai_name, c=ai_cfg: self._select_ai(n, c),
            )
        self._ai_drop.config(menu=self._ai_menu)
        self._ai_drop.pack(side="left")

        # ─ Separator ─
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4, pady=2)

        # ─ Group 4: Toggles & Settings ─
        self._use_diarization = tk.BooleanVar(value=True)
        tk.Checkbutton(
            top, text="👥", variable=self._use_diarization,
            font=("Segoe UI", 9), fg="#aaa", selectcolor="#1a1a2e",
        ).pack(side="left", padx=1)

        self._use_translate = tk.BooleanVar(value=True)
        self._use_translate.trace_add("write", self._on_translate_toggle)
        tk.Checkbutton(
            top, text="🌐", variable=self._use_translate,
            font=("Segoe UI", 9, "bold"), fg="#64b5f6", selectcolor="#1a1a2e",
        ).pack(side="left", padx=1)

        self.dual_btn = tk.Button(
            top, text="📖", width=2, command=self._toggle_dual,
            bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9),
        )
        self.dual_btn.pack(side="left", padx=1)

        tk.Button(top, text="A-", width=2, command=self._font_down,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side="left", padx=1)
        tk.Button(top, text="A+", width=2, command=self._font_up,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side="left", padx=1)

        tk.Button(top, text="⚙", width=2, command=self._open_terms_editor,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 9)).pack(side="left", padx=1)

        # ── Toolbar Row 2: Audio source + Device + Status ──────────────
        row2 = tk.Frame(root)
        row2.pack(fill="x", padx=8, pady=(0, 2))

        # Audio source toggle: Loopback (🔊) / Mic (🎙)
        self._audio_mode = tk.StringVar(value="loopback")
        tk.Radiobutton(
            row2, text="🔊 Loa", variable=self._audio_mode, value="loopback",
            font=("Segoe UI", 9), fg="#64b5f6", selectcolor="#1a1a2e",
            command=self._refresh_devices,
        ).pack(side="left")
        tk.Radiobutton(
            row2, text="🎙 Mic", variable=self._audio_mode, value="mic",
            font=("Segoe UI", 9), fg="#ffb74d", selectcolor="#1a1a2e",
            command=self._refresh_devices,
        ).pack(side="left", padx=(0, 4))

        self.devices = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or ["(Không tìm thấy device)"]
        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(row2, textvariable=self.dev_var,
                                      values=dev_names, state="readonly", width=42)
        self.dev_combo.pack(side="left", padx=4)
        default_idx = next(
            (i for i, n in enumerate(dev_names)
             if "Headphone" in n or "Realtek" in n or "Speaker" in n), 0
        )
        self.dev_combo.current(default_idx)

        self.status = tk.Label(row2, text="Chưa chạy", anchor="w",
                               fg="#555", font=("Segoe UI", 9))
        self.status.pack(side="left", padx=8, fill="x", expand=True)

        # ── Dual Panel: Source (trái) + Target (phải) ─────────────────
        self._panels = tk.PanedWindow(root, orient="horizontal",
                                       sashwidth=4, bg="#30363d")
        self._panels.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Panel Source (left)
        self._src_frame = tk.Frame(self._panels, bg="#0d1117")
        self._src_label = tk.Label(self._src_frame, text="", anchor="w",
                                    bg="#0d1117", fg="#58a6ff",
                                    font=("Segoe UI", 9, "bold"))
        self._src_label.pack(fill="x", padx=4)
        self.text_jp = scrolledtext.ScrolledText(
            self._src_frame, wrap="word", font=("Yu Gothic UI", self._font_size),
            bg="#111", fg="#eee", insertbackground="#eee", relief="flat",
        )
        self.text_jp.pack(fill="both", expand=True)
        self._panels.add(self._src_frame, stretch="always")

        # Panel Target (right)
        self._vi_frame = tk.Frame(self._panels, bg="#0d1117")
        self._tgt_label = tk.Label(self._vi_frame, text="", anchor="w",
                                    bg="#0d1117", fg="#3fb950",
                                    font=("Segoe UI", 9, "bold"))
        self._tgt_label.pack(fill="x", padx=4)
        self.text_vi = scrolledtext.ScrolledText(
            self._vi_frame, wrap="word", font=("Segoe UI", self._font_size),
            bg="#111", fg="#aff3c3", insertbackground="#aff3c3", relief="flat",
        )
        self.text_vi.pack(fill="both", expand=True)
        self._panels.add(self._vi_frame, stretch="always")

        self.text = self.text_jp
        self._update_panel_labels()

        # ── Phím tắt ──────────────────────────────────────────────────
        root.bind("<Control-Shift-C>", lambda e: self.copy_jp())
        root.bind("<Control-Shift-V>", lambda e: self.copy_vi())
        root.bind("<Control-Delete>",  lambda e: self.clear())
        root.bind("<Control-d>",       lambda e: self._toggle_dual())
        root.bind("<Control-t>",       lambda e: self._use_translate.set(not self._use_translate.get()))
        root.bind("<Control-g>",       lambda e: self._open_terms_editor())
        root.bind("<Control-Return>",  lambda e: self.ask_copilot())
        root.bind("<Control-Shift-Return>", lambda e: self.ask_claude())

        self.root.after(100, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ── Font control ──────────────────────────────────────────────────
    def _font_up(self):
        self._font_size = min(self._font_size + 2, 40)
        self._apply_font()

    def _font_down(self):
        self._font_size = max(self._font_size - 2, 8)
        self._apply_font()

    def _apply_font(self):
        self.text_jp.config(font=("Yu Gothic UI", self._font_size))
        self.text_vi.config(font=("Segoe UI", self._font_size))

    # ── Language helpers ──────────────────────────────────────────────
    def _get_src_lang(self) -> str:
        idx = self._src_combo.current()
        return self._lang_codes[idx] if idx >= 0 else "ja"

    def _get_tgt_lang(self) -> str:
        idx = self._tgt_combo.current()
        return self._lang_codes[idx] if idx >= 0 else "vi"

    def _get_lang_pair(self) -> tuple[str, str]:
        return self._get_src_lang(), self._get_tgt_lang()

    def _update_panel_labels(self):
        src = SUPPORTED_LANGS.get(self._get_src_lang(), {})
        tgt = SUPPORTED_LANGS.get(self._get_tgt_lang(), {})
        self._src_label.config(text=f"{src.get('flag','')} {src.get('name','Source')}")
        self._tgt_label.config(text=f"{tgt.get('flag','')} {tgt.get('name','Target')}")

    def _on_lang_change(self, event=None):
        self._update_panel_labels()
        src = self._get_src_lang()
        tgt = self._get_tgt_lang()
        self.root.title(f"Interview STT — {SUPPORTED_LANGS[src]['name']} → {SUPPORTED_LANGS[tgt]['name']}")

    # ── Dual / Single toggle ─────────────────────────────────────────
    def _toggle_dual(self):
        self._dual_mode = not self._dual_mode
        if self._dual_mode:
            self._panels.add(self._vi_frame, stretch="always")
            self.dual_btn.config(text="📖 Dual", bg="#21262d")
        else:
            self._panels.forget(self._vi_frame)
            self.dual_btn.config(text="📄 Single", bg="#21262d")

    # ── Translate toggle ──────────────────────────────────────────────
    def _on_translate_toggle(self, *_):
        self._translate_on = self._use_translate.get()
        if self._translate_on:
            self._ensure_translator()

    def _ensure_translator(self):
        if self.translator_thread is None or not self.translator_thread.is_alive():
            whisper_ready = self.transcriber.model_ready if self.transcriber else None
            self.translator_thread = TranslatorThread(
                self.jp_trans_queue, self.vi_queue, self.err_queue,
                status_queue=self.status_queue,
                wait_for_event=whisper_ready,
                get_lang_pair=self._get_lang_pair,
            )
            self.translator_thread.start()

    # ── Smart Scroll ──────────────────────────────────────────────────
    def _is_near_bottom(self, widget):
        try:
            _, yview_end = widget.yview()
            return yview_end > 0.95
        except Exception:
            return True

    def _smart_scroll(self, widget):
        if self._is_near_bottom(widget):
            widget.see("end")

    # ── Device ────────────────────────────────────────────────────────
    def _refresh_devices(self):
        mode = self._audio_mode.get()
        if mode == "mic":
            self.devices = list_mic_devices()
        else:
            self.devices = list_loopback_devices()
        dev_names = [n for _, n in self.devices] or [f"(Không tìm thấy {mode})"]
        self.dev_combo["values"] = dev_names
        if dev_names:
            self.dev_combo.current(0)

    def _selected_device_index(self):
        sel = self.dev_var.get()
        for idx, name in self.devices:
            if name == sel:
                return idx
        return None

    def _load_diarization(self):
        load_diarization_pipeline()
        if HAS_DIARIZATION:
            self.status_queue.put("Diarization OK. Đang nghe loa...")

    # ── Start / Stop ──────────────────────────────────────────────────
    def toggle(self):
        if not self.running:
            self.start()
        else:
            self.stop()

    def start(self):
        self.running = True
        self.btn.config(text="■ Stop", bg="#c62828")
        self.dev_combo.config(state="disabled")
        self._src_combo.config(state="disabled")
        self._tgt_combo.config(state="disabled")

        if not HAS_DIARIZATION and HF_TOKEN:
            self.set_status("Đang tải diarization model...")
            threading.Thread(target=self._load_diarization, daemon=True).start()

        if self._audio_mode.get() == "mic":
            self.audio = AudioMic(self.frame_queue,
                                  device_index=self._selected_device_index())
        else:
            self.audio = AudioLoopback(self.frame_queue,
                                       device_index=self._selected_device_index())
        try:
            self.audio.start()
        except Exception as e:
            self.set_status(f"Lỗi audio: {e}")
            self.running = False
            self.btn.config(text="▶ Start", bg="#2e7d32")
            self.dev_combo.config(state="readonly")
            self._src_combo.config(state="readonly")
            self._tgt_combo.config(state="readonly")
            return

        self.transcriber = Transcriber(
            self.frame_queue, self.text_queue, self.status_queue,
            lambda: (self.audio.device_rate, self.audio.channels),
            use_diarization=self._use_diarization,
            get_src_lang=self._get_src_lang,
        )
        self.transcriber.start()

        # Auto-start translator nếu dịch đang bật
        if self._translate_on:
            self._ensure_translator()

    def stop(self):
        self.running = False
        self.btn.config(text="▶ Start", bg="#2e7d32")
        self.dev_combo.config(state="readonly")
        self._src_combo.config(state="readonly")
        self._tgt_combo.config(state="readonly")
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
        self._last_speaker = None
        _speaker_registry.reset()
        self.set_status("Đã dừng")

    # ── Clear / Copy ──────────────────────────────────────────────────
    def clear(self):
        self.text_jp.delete("1.0", "end")
        self.text_vi.delete("1.0", "end")
        self._pending_vi.clear()
        self._seg_counter = 0
        self._vi_next_id = 0

    def copy_jp(self):
        content = self.text_jp.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._flash_status("✅ Đã copy JP!")

    def copy_vi(self):
        content = self.text_vi.get("1.0", "end").strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self._flash_status("✅ Đã copy VI!")

    # ── AI automation (generic) ──────────────────────────────────────
    _AI_REGISTRY = {
        "Copilot":  (open_or_focus_copilot, find_copilot_hwnd, click_copilot_input, 4500),
        "Claude":   (open_or_focus_claude,  find_claude_hwnd,  click_claude_input,  5000),
        "ChatGPT":  (open_or_focus_chatgpt, find_chatgpt_hwnd, click_chatgpt_input, 4500),
    }

    def _select_ai(self, name, cfg):
        self._ai_choice = name
        self._ai_send_btn.config(text=f"🤖 {name}", bg=cfg["color"])
        self._ai_drop.config(bg=cfg["color"], activebackground=cfg["color"])

    def _send_to_ai(self):
        reg = self._AI_REGISTRY.get(self._ai_choice)
        if reg:
            open_fn, find_fn, click_fn, delay = reg
            self._ask_ai(self._ai_choice, open_fn, find_fn, click_fn, delay)

    def ask_copilot(self):
        self._select_ai("Copilot", {"color": "#0078d4", "icon": "🟦"})
        self._send_to_ai()

    def ask_claude(self):
        self._select_ai("Claude", {"color": "#d97706", "icon": "🟧"})
        self._send_to_ai()

    def ask_chatgpt(self):
        self._select_ai("ChatGPT", {"color": "#10a37f", "icon": "🟩"})
        self._send_to_ai()

    def _ask_ai(self, name, open_fn, find_fn, click_fn, launch_delay=4500):
        content = self.text_jp.get("1.0", "end").strip()
        if not content:
            self._flash_status("⚠️ Chưa có text nào để hỏi!")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        hwnd = open_fn()
        self._ai_target = (name, find_fn, click_fn)
        if hwnd:
            self._flash_status(f"⏳ Đang gửi sang {name}...")
            self.root.after(400, self._ai_click_and_paste)
        else:
            self._flash_status(f"🚀 Đang mở {name}...")
            self.root.after(launch_delay, self._ai_click_and_paste)

    def _ai_click_and_paste(self):
        name, find_fn, click_fn = self._ai_target
        hwnd = find_fn()
        if hwnd:
            click_fn(hwnd)
        self.root.after(350, self._ai_do_paste)

    def _ai_do_paste(self):
        user32 = ctypes.windll.user32
        KEYUP = 0x0002
        user32.keybd_event(0x11, 0, 0, 0)
        user32.keybd_event(0x56, 0, 0, 0)
        user32.keybd_event(0x56, 0, KEYUP, 0)
        user32.keybd_event(0x11, 0, KEYUP, 0)
        self.root.after(300, self._ai_do_enter)

    def _ai_do_enter(self):
        user32 = ctypes.windll.user32
        KEYUP = 0x0002
        user32.keybd_event(0x0D, 0, 0, 0)
        user32.keybd_event(0x0D, 0, KEYUP, 0)
        name = self._ai_target[0]
        self._flash_status(f"✅ Đã gửi! {name} đang trả lời...")

    def ask_copilot(self):
        self._ask_ai("Copilot", open_or_focus_copilot,
                      find_copilot_hwnd, click_copilot_input)

    def ask_claude(self):
        self._ask_ai("Claude", open_or_focus_claude,
                      find_claude_hwnd, click_claude_input, launch_delay=5000)

    def ask_chatgpt(self):
        self._ask_ai("ChatGPT", open_or_focus_chatgpt,
                      find_chatgpt_hwnd, click_chatgpt_input)

    # ── Status helpers ────────────────────────────────────────────────
    def _flash_status(self, msg):
        old = self.status.cget("text")
        self.status.config(text=msg, fg="#00e676")
        self.root.after(2000, lambda: self.status.config(text=old, fg="#555"))

    def set_status(self, msg):
        self.status.config(text=msg, fg="#555")

    # ── Poll queues ───────────────────────────────────────────────────
    def poll(self):
        while not self.status_queue.empty():
            self.set_status(self.status_queue.get())

        # JP text từ Whisper
        while not self.text_queue.empty():
            item    = self.text_queue.get()
            jp_text = item["text"].strip()
            color   = item.get("color", "#eee")
            speaker = item.get("speaker")

            tag = f"spk_{color.replace('#','')}"
            self.text_jp.tag_configure(tag, foreground=color)

            same_speaker = (speaker == self._last_speaker)
            if not same_speaker:
                if self._last_speaker is not None:
                    self.text_jp.insert("end", "\n", tag)
                if speaker:
                    self.text_jp.insert("end", f"{speaker}\n", tag)
                self.text_jp.insert("end", jp_text, tag)
                self._last_speaker = speaker
            else:
                last_char = self.text_jp.get("end-2c", "end-1c")
                if last_char in ("。", "！", "？", ".", "!", "?"):
                    self.text_jp.insert("end", "\n" + jp_text, tag)
                else:
                    self.text_jp.insert("end", jp_text, tag)

            self._smart_scroll(self.text_jp)

            # FIFO: gửi (seg_id, jp_text) qua translator
            if self._translate_on:
                seg_id = self._seg_counter
                self._seg_counter += 1
                self.jp_trans_queue.put((seg_id, jp_text))
                # Hiện placeholder "⏳" trong panel VI
                self.text_vi.insert("end", "⏳ ...\n", "pending")
                self.text_vi.tag_configure("pending", foreground="#555")
                self._pending_vi[seg_id] = self.text_vi.index("end-2l linestart")
                self._smart_scroll(self.text_vi)

        # VI text từ Qwen / Google — FIFO thay thế placeholder
        while not self.vi_queue.empty():
            result = self.vi_queue.get()
            if isinstance(result, tuple):
                seg_id, vi_text = result
            else:
                seg_id, vi_text = None, result

            if seg_id is not None and seg_id in self._pending_vi:
                # Thay placeholder bằng bản dịch thật
                pos = self._pending_vi.pop(seg_id)
                try:
                    line_end = f"{pos} lineend"
                    self.text_vi.delete(pos, line_end)
                    self.text_vi.insert(pos, vi_text, "translated")
                    self.text_vi.tag_configure("translated", foreground="#aff3c3")
                except Exception:
                    self.text_vi.insert("end", vi_text + "\n", "translated")
            else:
                self.text_vi.insert("end", vi_text + "\n", "translated")
                self.text_vi.tag_configure("translated", foreground="#aff3c3")

            self._smart_scroll(self.text_vi)

        # Lỗi dịch
        while not self.err_queue.empty():
            err = self.err_queue.get()
            self.set_status(f"⚠️ {err}")

        self.root.after(100, self.poll)

    # ── Terms Editor ────────────────────────────────────────────────
    def _open_terms_editor(self):
        win = tk.Toplevel(self.root)
        win.title("⚙ Thuật ngữ phỏng vấn (JP → VI)")
        win.geometry("500x420")
        win.attributes("-topmost", True)

        tk.Label(win, text="Mỗi dòng: 日本語 = Tiếng Việt",
                 font=("Segoe UI", 10), fg="#888").pack(padx=10, pady=(8, 2), anchor="w")

        txt = scrolledtext.ScrolledText(
            win, wrap="word", font=("Segoe UI", 12),
            bg="#111", fg="#eee", insertbackground="#eee",
        )
        txt.pack(fill="both", expand=True, padx=10, pady=4)

        # Load terms hiện tại
        global _custom_terms
        for jp, vi in _custom_terms.items():
            txt.insert("end", f"{jp} = {vi}\n")

        status_lbl = tk.Label(win, text="", fg="#888", font=("Segoe UI", 9))
        status_lbl.pack(padx=10, anchor="w")

        def save():
            global _custom_terms
            content = txt.get("1.0", "end").strip()
            new_terms = {}
            for line in content.splitlines():
                line = line.strip()
                if "=" in line:
                    parts = line.split("=", 1)
                    jp = parts[0].strip()
                    vi = parts[1].strip()
                    if jp and vi:
                        new_terms[jp] = vi
            _custom_terms = new_terms
            try:
                with open(TERMS_PATH, "w", encoding="utf-8") as f:
                    json.dump(_custom_terms, f, ensure_ascii=False, indent=2)
            except Exception as e:
                status_lbl.config(text=f"❌ Lỗi lưu: {e}", fg="#ef5350")
                return
            # Cập nhật hint trong Qwen translator nếu đang chạy
            if (self.translator_thread and hasattr(self.translator_thread, '_qwen')
                    and self.translator_thread._qwen):
                self.translator_thread._qwen._terms_hint = _build_terms_hint()
            status_lbl.config(text=f"✅ Đã lưu {len(new_terms)} thuật ngữ!", fg="#3fb950")
            win.after(2000, lambda: status_lbl.config(text="", fg="#888"))

        btn_bar = tk.Frame(win)
        btn_bar.pack(fill="x", padx=10, pady=(0, 8))
        tk.Button(btn_bar, text="💾 Lưu", width=10, command=save,
                  bg="#238636", fg="white", font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Button(btn_bar, text="Đóng", width=8, command=win.destroy,
                  bg="#21262d", fg="#c9d1d9", font=("Segoe UI", 10)).pack(side="left", padx=6)
        tk.Label(btn_bar, text=f"📁 {TERMS_PATH}",
                 fg="#555", font=("Segoe UI", 8)).pack(side="right")

    def on_close(self):
        self.stop()
        if self.translator_thread:
            self.translator_thread.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
