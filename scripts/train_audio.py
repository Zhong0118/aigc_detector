"""Audio AI detection model training script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa

from detection.audio_detector import AudioCNN


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, sr=16000, duration=5):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        y, _ = librosa.load(str(self.audio_paths[idx]), sr=self.sr, duration=self.duration)

        target_len = self.sr * self.duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        tensor = torch.FloatTensor(mel_db).unsqueeze(0)

        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = Path("data/audio")
    if not data_dir.exists():
        print("No training data found in data/audio/. Please prepare your dataset.")
        print("Expected structure: data/audio/real/*.wav and data/audio/ai/*.wav")
        return

    real_audio = list((data_dir / "real").glob("*.*"))
    ai_audio = list((data_dir / "ai").glob("*.*"))
    audio_paths = real_audio + ai_audio
    labels = [0] * len(real_audio) + [1] * len(ai_audio)

    dataset = AudioDataset(audio_paths, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0

        for mels, targets in loader:
            mels, targets = mels.to(device), targets.to(device)
            logits = model(mels)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}, Acc: {acc:.4f}")

    Path("data/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "data/models/audio_classifier.pt")
    print("Model saved to data/models/audio_classifier.pt")


if __name__ == "__main__":
    train()
