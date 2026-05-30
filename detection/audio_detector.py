"""本地音频 AIGC/Deepfake 检测 scaffold。

当前不是 API-first 主流程的默认路径。
这里提供一个简单的 mel spectrogram + CNN 二分类结构，
用于后续替换成 AASIST、DeepFense、RawNet、WavLM 等更成熟模型。

注意：
- 当前模型没有训练权重，不能作为真实判断依据。
- 真正使用前需要 load_weights 或接入预训练 checkpoint。
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import librosa


@dataclass
class DetectionResult:
    """本地音频 detector 的统一返回结构。"""

    score: float
    label: str
    details: dict


class AudioCNN(nn.Module):
    """简单音频 CNN。

    输入是 mel spectrogram，输出是 human/ai 二分类 logits。
    """

    def __init__(self, n_mels: int = 128):
        """构建卷积特征提取器和分类头。"""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        """前向传播。"""
        x = self.features(x)
        return self.classifier(x)


class AudioDetector:
    """音频检测器 scaffold。"""

    def __init__(self, sample_rate: int = 16000, device: str | None = None):
        """初始化采样率、设备和 CNN 模型。"""
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioCNN().to(self.device)
        self.model.eval()

    def detect(self, audio: np.ndarray) -> DetectionResult:
        """对音频数组进行检测。

        audio 来自 ingestion.loader 中 librosa.load 的结果。
        """
        mel_spec = self._compute_mel_spectrogram(audio)
        spectral_features = self._extract_spectral_features(audio)

        tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0, 1].item()

        label = "ai" if ai_prob > 0.5 else "human"
        return DetectionResult(
            score=ai_prob,
            label=label,
            details={"model_prob": ai_prob, **spectral_features},
        )

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """计算 mel spectrogram，并转换为 dB 尺度。"""
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=128, fmax=8000
        )
        return librosa.power_to_db(mel, ref=np.max)

    def _extract_spectral_features(self, audio: np.ndarray) -> dict:
        """提取简单频谱特征。

        这些特征可以作为解释信息，也可以给后续融合/归因模型使用。
        """
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        zcr = librosa.feature.zero_crossing_rate(audio)

        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
        }

    def load_weights(self, path: str):
        """加载训练好的音频检测模型权重。"""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
