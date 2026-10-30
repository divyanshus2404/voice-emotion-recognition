"""Core logic: load a wav2vec2 emotion model and run it on audio.

Pipeline in one line:  raw audio -> 16 kHz mono -> wav2vec2 -> softmax -> emotion scores
"""
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SAMPLE_RATE = 16_000  # wav2vec2 was pretrained on 16 kHz speech
LOCAL_MODEL = Path(__file__).resolve().parent.parent / "model"  # created by train.py
PRETRAINED_MODEL = "superb/wav2vec2-base-superb-er"  # 4 emotions, trained on IEMOCAP

# Short codes used by some checkpoints -> readable names
LABEL_NAMES = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}

EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


class EmotionPredictor:
    def __init__(self, model_name: str | None = None):
        name = model_name or (str(LOCAL_MODEL) if LOCAL_MODEL.exists() else PRETRAINED_MODEL)
        self.model_name = name
        self.extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModelForAudioClassification.from_pretrained(name).eval()
        self.labels = [
            LABEL_NAMES.get(self.model.config.id2label[i], self.model.config.id2label[i])
            for i in range(len(self.model.config.id2label))
        ]

    @staticmethod
    def prepare(audio: np.ndarray, sr: int) -> np.ndarray:
        """Convert any audio to the format the model expects: float32, mono, 16 kHz, normalised."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:  # stereo -> mono
            audio = audio.mean(axis=1 if audio.shape[1] <= 2 else 0)
        if np.abs(audio).max() > 1.0:  # int16 from the microphone
            audio = audio / 32768.0
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio, _ = librosa.effects.trim(audio, top_db=30)  # cut leading/trailing silence
        peak = np.abs(audio).max()
        return audio / peak if peak > 0 else audio

    @torch.inference_mode()
    def scores(self, audio16k: np.ndarray) -> dict[str, float]:
        inputs = self.extractor(audio16k, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        probs = torch.softmax(self.model(**inputs).logits[0], dim=-1)
        return {label: float(p) for label, p in zip(self.labels, probs)}

    def predict(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Emotion scores for the whole clip."""
        return self.scores(self.prepare(audio, sr))

    def timeline(self, audio: np.ndarray, sr: int, window_s: float = 2.0, hop_s: float = 1.0):
        """Emotion scores for overlapping windows, so you can see the mood change over time."""
        audio = self.prepare(audio, sr)
        win, hop = int(window_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
        if len(audio) <= win:
            return [(0.0, self.scores(audio))]
        return [
            (start / SAMPLE_RATE, self.scores(audio[start:start + win]))
            for start in range(0, len(audio) - win + 1, hop)
        ]
