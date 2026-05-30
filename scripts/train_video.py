"""视频 AIGC 检测模型训练脚本。

这个脚本属于后续“本地模型路线”，不是当前 API-first MVP 的必要步骤。
当前内容是训练 scaffold：
- 从视频中抽帧
- 用 EfficientNetV2 提取每帧特征
- 用 LSTM 做时序聚合
- 输出 human/ai 二分类

后续需要补充：
- 使用真实 AI video 数据集
- 更稳定的抽帧策略和视频解码错误处理
- train/val/test 切分
- 指标评估：AUC、F1、帧级/视频级准确率
- 保存 checkpoint metadata
- 可替换为 cakelens、D3、VideoSeal 或专用视频模型
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from PIL import Image


class VideoDataset(Dataset):
    """视频训练数据集。

    video_paths 是视频路径，labels 中 0 表示 real，1 表示 ai。
    """

    def __init__(self, video_paths, labels, max_frames=16, frame_interval=30):
        """保存抽帧配置和图片 transform。"""
        self.video_paths = video_paths
        self.labels = labels
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """返回样本数量。"""
        return len(self.video_paths)

    def __getitem__(self, idx):
        """读取一个视频，抽帧，补齐/截断到固定帧数。"""
        frames = self._extract_frames(self.video_paths[idx])

        if len(frames) < self.max_frames:
            # 帧数不足时用全零帧补齐。
            padding = [torch.zeros(3, 224, 224)] * (self.max_frames - len(frames))
            frames.extend(padding)
        else:
            frames = frames[:self.max_frames]

        tensor = torch.stack(frames)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)

    def _extract_frames(self, path):
        """从视频中按固定间隔抽取帧并转成 tensor。"""
        cap = cv2.VideoCapture(str(path))
        frames = []
        idx = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.frame_interval == 0:
                # OpenCV 读取是 BGR，需要转成 RGB 后交给 PIL/transform。
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(self.transform(pil))
            idx += 1
        cap.release()
        return frames


class VideoClassifier(nn.Module):
    """视频二分类模型。

    每帧用 EfficientNetV2 提特征，再用双向 LSTM 聚合时序信息。
    """

    def __init__(self, max_frames=16):
        """构建帧特征提取器、LSTM 和分类头。"""
        super().__init__()
        backbone = models.efficientnet_v2_s(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.temporal = nn.LSTM(1280, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """前向传播。

        x shape: batch, time, channel, height, width
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        features = features.view(b, t, -1)
        lstm_out, _ = self.temporal(features)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)


def train():
    """训练视频二分类模型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = Path("data/videos")
    if not data_dir.exists():
        print("No training data found in data/videos/. Please prepare your dataset.")
        print("Expected structure: data/videos/real/*.mp4 and data/videos/ai/*.mp4")
        return

    real_videos = list((data_dir / "real").glob("*.*"))
    ai_videos = list((data_dir / "ai").glob("*.*"))
    video_paths = real_videos + ai_videos
    labels = [0] * len(real_videos) + [1] * len(ai_videos)

    dataset = VideoDataset(video_paths, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = VideoClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for videos, targets in loader:
            videos, targets = videos.to(device), targets.to(device)
            logits = model(videos)
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
    torch.save(model.state_dict(), "data/models/video_classifier.pt")
    print("Model saved to data/models/video_classifier.pt")


if __name__ == "__main__":
    # 直接运行脚本时启动训练。
    train()
