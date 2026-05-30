"""本地图片 AIGC 检测 scaffold。

当前不是 API-first 主流程的默认路径。
这里搭了一个 torchvision 模型结构和若干像素统计特征，
用于后续接入真实训练权重或替换成 Hugging Face 图片检测模型。

注意：
- 现在模型 weights=None，相当于未训练模型，不能作为真实判断依据。
- 真正使用前需要 load_weights 或替换为已训练 checkpoint。
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


@dataclass
class DetectionResult:
    """本地图片 detector 的统一返回结构。"""

    score: float
    label: str
    details: dict


class ImageDetector:
    """图片检测器 scaffold。

    默认结构是 EfficientNetV2-S 二分类，也可回退到 ResNet18。
    """

    def __init__(self, model_name: str = "efficientnet_v2_s", device: str | None = None):
        """初始化模型、设备和图像预处理 transform。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self._build_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self) -> nn.Module:
        """构建二分类模型结构。

        输出类别约定为：
        index 0 -> human
        index 1 -> ai
        """
        if self.model_name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        return model.to(self.device)

    def detect(self, image: Image.Image) -> DetectionResult:
        """对 PIL 图片进行本地检测。

        当前模型未训练时，model_prob 不具备真实意义；
        pixel_stats 可作为辅助特征输出。
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0, 1].item()

        pixel_stats = self._analyze_pixel_artifacts(image)

        label = "ai" if ai_prob > 0.5 else "human"
        return DetectionResult(
            score=ai_prob,
            label=label,
            details={"model_prob": ai_prob, **pixel_stats},
        )

    def _analyze_pixel_artifacts(self, image: Image.Image) -> dict:
        """提取简单像素统计特征。

        后续可扩展为 GAN artifacts、频域特征、patch consistency 等。
        """
        arr = np.array(image).astype(np.float32)
        return {
            "mean_intensity": float(arr.mean()),
            "std_intensity": float(arr.std()),
            "high_freq_energy": self._high_freq_energy(arr),
        }

    def _high_freq_energy(self, arr: np.ndarray) -> float:
        """计算高频能量。

        频域特征有时可以帮助观察生成图像中的纹理/伪影模式。
        """
        gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        mask = np.ones_like(gray, dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = False
        high_freq = np.abs(fft_shift[mask])
        return float(np.mean(high_freq))

    def load_weights(self, path: str):
        """加载训练好的模型权重。

        后续如果训练了 EfficientNet/ResNet 二分类模型，可以通过这个方法加载。
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
