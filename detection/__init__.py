"""detection 包的统一导出入口。

detection 层负责“AI vs Human 检测”。
当前项目有两类检测路线：

1. API-first 路线：
   - 当前主流程使用 `detection/providers.py`
   - 默认跑 demo_api
   - 后续可接 Hive、Sightengine 或其他检测 API

2. 本地模型路线：
   - `text_detector.py`
   - `image_detector.py`
   - `audio_detector.py`
   - `video_detector.py`
   这些是本地模型 scaffold，不是当前默认生产检测路径。

这里使用 `__getattr__` 做懒加载，避免导入 detection 包时立刻加载
torch、transformers、torchvision、librosa、cv2 等重依赖。
"""

def __getattr__(name):
    """按需导出本地 detector 和 fusion 工具。

    外部可以写 `from detection import TextDetector`，
    Python 会在真正访问时才加载对应模块。
    """
    if name == "TextDetector":
        # 本地文本检测 scaffold，依赖 transformers。
        from .text_detector import TextDetector
        return TextDetector
    if name == "ImageDetector":
        # 本地图片检测 scaffold，依赖 torch/torchvision/PIL。
        from .image_detector import ImageDetector
        return ImageDetector
    if name == "AudioDetector":
        # 本地音频检测 scaffold，依赖 torch/librosa。
        from .audio_detector import AudioDetector
        return AudioDetector
    if name == "VideoDetector":
        # 本地视频检测 scaffold，内部复用 ImageDetector。
        from .video_detector import VideoDetector
        return VideoDetector
    if name == "FusionClassifier":
        # 多个分数融合工具。
        from .fusion import FusionClassifier
        return FusionClassifier
    raise AttributeError(f"module 'detection' has no attribute {name!r}")
