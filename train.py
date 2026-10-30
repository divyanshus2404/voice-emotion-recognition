"""Fine-tune wav2vec2 on RAVDESS (8 emotions). Run on a GPU (Google Colab free tier works).

    python train.py --data_dir path/to/RAVDESS --epochs 10

Download RAVDESS speech audio (Audio_Speech_Actors_01-24.zip) from https://zenodo.org/records/1188976
The trained model is saved to ./model and app.py picks it up automatically.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SR = 16_000
MAX_SECONDS = 4
# RAVDESS file name: 03-01-05-01-02-01-12.wav -> the 3rd number is the emotion
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
LABELS = list(EMOTIONS.values())


class Ravdess(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        audio, _ = librosa.load(f, sr=SR, duration=MAX_SECONDS)
        audio = np.pad(audio, (0, SR * MAX_SECONDS - len(audio)))
        return audio, LABELS.index(EMOTIONS[f.name.split("-")[2]])


def collate(batch, extractor):
    audios, labels = zip(*batch)
    inputs = extractor(list(audios), sampling_rate=SR, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--base_model", default="facebook/wav2vec2-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.wav"))
    # Split by ACTOR, not by file: the model must never hear a test speaker during training.
    actor = lambda f: int(f.stem.split("-")[-1])
    train_files = [f for f in files if actor(f) <= 20]
    test_files = [f for f in files if actor(f) > 20]
    print(f"{len(train_files)} train clips (actors 1-20), {len(test_files)} test clips (actors 21-24)")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(args.base_model)
    model = AutoModelForAudioClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)
    model.freeze_feature_encoder()  # keep the low-level CNN frozen: faster and less overfitting

    col = lambda b: collate(b, extractor)
    train_dl = DataLoader(Ravdess(train_files), batch_size=args.batch_size, shuffle=True, collate_fn=col)
    test_dl = DataLoader(Ravdess(test_files), batch_size=args.batch_size, collate_fn=col)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(train_dl):.3f}")

    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_dl:
            gold += batch.pop("labels").tolist()
            preds += model(**{k: v.to(device) for k, v in batch.items()}).logits.argmax(-1).cpu().tolist()
    print(classification_report(gold, preds, target_names=LABELS, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):\n", confusion_matrix(gold, preds))

    model.save_pretrained("model")
    extractor.save_pretrained("model")
    print("Saved to ./model — restart app.py to use it.")


if __name__ == "__main__":
    main()
