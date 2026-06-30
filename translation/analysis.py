# -*- coding: utf-8 -*-
"""
AI Analysis — dùng Qwen để tóm tắt, trích keywords, phát hiện vấn đề.
"""

import queue
import threading

from config import LANG_NAMES_EN
from translation.qwen import load_qwen_translator

ANALYSIS_MODES = {
    "summary": {
        "label": "📝 Summary",
        "prompt": (
            "Summarize the following conversation concisely in {tgt_name}. "
            "Focus on key points, decisions, and action items. "
            "Reply with ONLY the summary, no extra explanation."
        ),
    },
    "keywords": {
        "label": "🔑 Keywords",
        "prompt": (
            "Extract the most important keywords and technical terms "
            "from the following conversation. "
            "Reply as a bullet list in {tgt_name}. "
            "Format: • keyword — brief explanation"
        ),
    },
    "issues": {
        "label": "⚠️ Issues",
        "prompt": (
            "Analyze the following conversation and identify: "
            "difficult questions, contradictions, unclear points, or potential issues. "
            "Reply as a bullet list in {tgt_name}."
        ),
    },
    "answer": {
        "label": "💡 Suggestions",
        "prompt": (
            "Based on the following interview conversation, suggest a good answer "
            "for the last question asked. Reply in {tgt_name}. "
            "Be concise and professional."
        ),
    },
}


class AnalysisThread(threading.Thread):
    def __init__(self, result_queue, status_queue, text, mode, tgt_lang="vi",
                 qwen_instance=None):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.status_queue = status_queue
        self.text = text
        self.mode = mode
        self.tgt_lang = tgt_lang
        self._qwen = qwen_instance

    def run(self):
        import torch, re

        if not self._qwen:
            self.status_queue.put("⏳ Loading Qwen for analysis...")
            self._qwen = load_qwen_translator()
            if not self._qwen:
                self.result_queue.put(("error", "❌ Failed to load Qwen model"))
                return

        cfg = ANALYSIS_MODES.get(self.mode)
        if not cfg:
            self.result_queue.put(("error", f"Unknown mode: {self.mode}"))
            return

        tgt_name = LANG_NAMES_EN.get(self.tgt_lang, "Vietnamese")
        system_msg = cfg["prompt"].format(tgt_name=tgt_name)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": self.text},
        ]

        try:
            self.status_queue.put(f"🧠 Analyzing ({cfg['label']})...")
            prompt = self._qwen.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._qwen.tokenizer(prompt, return_tensors="pt").to(self._qwen._device)
            with torch.no_grad():
                out = self._qwen.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            result = self._qwen.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            self.result_queue.put(("result", result))
            self.status_queue.put(f"✅ Analysis done ({cfg['label']})")
        except Exception as e:
            self.result_queue.put(("error", f"Analysis error: {e}"))
            self.status_queue.put(f"❌ Analysis error: {e}")
