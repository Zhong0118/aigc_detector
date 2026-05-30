"""本地视频 AIGC 检测 scaffold。

当前不是 API-first 主流程的默认路径。
这里的思路是：从视频中抽帧，再复用图片检测器对每帧打分，
最后对帧分数做平均和时序稳定性统计。

注意：
- 这只是一个视频检测 baseline。
- 真正的视频生成检测后续可以接 cakelens、D3、VideoSeal 或专门的视频 API。
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import cv2
from PIL import Image

from .image_detector import ImageDetector


@dataclass
class DetectionResult:
    """本地视频 detector 的统一返回结构。"""

    score: float
    label: str
    details: dict


class VideoDetector:
    """基于抽帧 + 图片检测器的视频检测 scaffold。"""

    def __init__(self, frame_interval: int = 30, device: str | None = None):
        """初始化抽帧间隔和内部图片检测器。

        frame_interval=30 表示每 30 帧取一帧。
        """
        self.frame_interval = frame_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_detector = ImageDetector(device=self.device)

    def detect(self, video_path: str) -> DetectionResult:
        """分析视频文件。

        返回平均帧分数、帧数量和帧分数标准差。
        """
        frames = self._extract_frames(video_path)
        if not frames:
            return DetectionResult(score=0.0, label="unknown", details={"error": "no frames extracted"})

        frame_scores = []
        for frame in frames:
            # 当前视频检测复用 ImageDetector；后续可替换为视频专用模型。
            result = self.image_detector.detect(frame)
            frame_scores.append(result.score)

        avg_score = float(np.mean(frame_scores))
        temporal_consistency = float(np.std(frame_scores))

        label = "ai" if avg_score > 0.5 else "human"
        return DetectionResult(
            score=avg_score,
            label=label,
            details={
                "frame_count": len(frames),
                "avg_frame_score": avg_score,
                "temporal_consistency": temporal_consistency,
                "frame_scores": frame_scores,
            },
        )

    def _extract_frames(self, video_path: str) -> List[Image.Image]:
        """从视频中按固定间隔抽取 PIL 图片帧。"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_interval == 0:
                # OpenCV 帧是 BGR，转成 RGB 后交给 PIL/ImageDetector。
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil_img)
            frame_idx += 1

        cap.release()
        return frames

    def load_weights(self, path: str):
        """给内部图片检测器加载权重。

        当前视频 detector 的权重加载等价于加载 frame-level image detector 权重。
        """
        self.image_detector.load_weights(path)
