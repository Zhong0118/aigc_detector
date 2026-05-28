from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import cv2
from PIL import Image

from .image_detector import ImageDetector


@dataclass
class DetectionResult:
    score: float
    label: str
    details: dict


class VideoDetector:
    def __init__(self, frame_interval: int = 30, device: str | None = None):
        self.frame_interval = frame_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_detector = ImageDetector(device=self.device)

    def detect(self, video_path: str) -> DetectionResult:
        frames = self._extract_frames(video_path)
        if not frames:
            return DetectionResult(score=0.0, label="unknown", details={"error": "no frames extracted"})

        frame_scores = []
        for frame in frames:
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
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_interval == 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil_img)
            frame_idx += 1

        cap.release()
        return frames

    def load_weights(self, path: str):
        self.image_detector.load_weights(path)
