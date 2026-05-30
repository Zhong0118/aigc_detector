"""音频 AIGC/Deepfake 检测模型训练脚本。

这个脚本属于后续“本地模型路线”，不是当前 API-first MVP 的必要步骤。
当前内容是训练 scaffold：
- 读取 data/audio/real 和 data/audio/ai
- 把音频转成 mel spectrogram
- 训练 detection/audio_detector.py 中定义的 AudioCNN

后续需要补充：
- 使用真实 audio deepfake 数据集
- 支持不同格式、时长切片和静音过滤
- train/val/test 切分
- 指标评估：EER、AUC、F1、召回率
- 保存 checkpoint metadata
- 可替换为 AASIST、WavLM、DeepFense 等模型
"""
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
    """音频训练数据集。

    audio_paths 是音频文件路径，labels 中 0 表示 real，1 表示 ai/deepfake。
    """

    def __init__(self, audio_paths, labels, sr=16000, duration=5):
        """保存采样率和固定音频长度。"""
        self.audio_paths = audio_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration

    def __len__(self):
        """返回样本数量。"""
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """读取音频、裁剪/补齐长度，并转换为 mel spectrogram tensor。"""
        y, _ = librosa.load(str(self.audio_paths[idx]), sr=self.sr, duration=self.duration)

        target_len = self.sr * self.duration
        if len(y) < target_len:
            # 不足固定长度时补零。
            y = np.pad(y, (0, target_len - len(y)))
        else:
            # 超过固定长度时截断。
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        tensor = torch.FloatTensor(mel_db).unsqueeze(0)

        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)


def train():
    """训练音频二分类模型。"""
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
    # 直接运行脚本时启动训练。
    train()
