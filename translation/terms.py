# -*- coding: utf-8 -*-
"""
Custom terms (glossary) load/save and hint builder.
"""

import os
import json

TERMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terms.json")
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
