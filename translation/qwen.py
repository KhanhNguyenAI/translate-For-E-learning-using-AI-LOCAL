# -*- coding: utf-8 -*-
"""
QwenTranslator, load_qwen_translator, TranslatorThread.
"""

import queue
import threading

from config import QWEN_MODEL_NAME, LANG_NAMES_EN, FEWSHOT_EXAMPLES
from translation.terms import _build_terms_hint

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False
    print("[WARN] deep-translator not installed — Google Translate fallback disabled")

HAS_QWEN = False


class QwenTranslator:
    """Multi-language translator using Qwen 3 1.7B on local GPU."""

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


class TranslatorThread(threading.Thread):
    """Receives text from queue, translates via Qwen (primary) or Google Translate (fallback)."""

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
            self._set_status("⏳ Waiting for STT model to load...")
            self._wait_for.wait(timeout=120)

        self._set_status("\U0001f916 Loading Qwen 3 translator...")
        self._qwen = load_qwen_translator()
        if self._qwen:
            self.engine_name = "Qwen 3 1.7B"
            self._set_status("\U0001f916 Translator: Qwen 3 1.7B (local GPU)")
        else:
            if HAS_TRANSLATOR:
                src, tgt = self._get_lang_pair()
                try:
                    self._google = GoogleTranslator(source=src, target=tgt)
                    self._google_pair = (src, tgt)
                    self.engine_name = "Google Translate"
                    self._set_status("\U0001f310 Translator: Google Translate (fallback)")
                except Exception as e:
                    self.err_queue.put(f"Failed to initialize translator: {e}")
                    return
            else:
                self.err_queue.put("No translation engine available")
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
                self.err_queue.put(f"Translation error ({self.engine_name}): {e}")

    def stop(self):
        self._stop.set()
