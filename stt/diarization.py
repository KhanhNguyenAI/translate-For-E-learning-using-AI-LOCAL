# -*- coding: utf-8 -*-
"""
SpeakerRegistry, diarize_audio, load_diarization_pipeline.
"""

import numpy as np

from config import TARGET_SR, HF_TOKEN

_diarize_pipeline = None
HAS_DIARIZATION = False

# Màu theo speaker index
SPEAKER_COLORS = ["#64b5f6", "#81c784", "#ffb74d", "#f06292", "#ba68c8"]
SPEAKER_LABELS = ["\U0001f399 Speaker A", "\U0001f399 Speaker B", "\U0001f399 Speaker C",
                  "\U0001f399 Speaker D", "\U0001f399 Speaker E"]


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


class SpeakerRegistry:
    """Theo dõi speaker identity xuyên suốt các chunk bằng embedding similarity."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.profiles  = {}   # global_label -> embedding tensor
        self.next_idx  = 0

    def resolve(self, local_id: str, embeddings: dict) -> str:
        """Map local speaker id (SPEAKER_00...) -> global label nhất quán."""
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
