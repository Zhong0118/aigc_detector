def __getattr__(name):
    if name == "TextDetector":
        from .text_detector import TextDetector
        return TextDetector
    if name == "ImageDetector":
        from .image_detector import ImageDetector
        return ImageDetector
    if name == "AudioDetector":
        from .audio_detector import AudioDetector
        return AudioDetector
    if name == "VideoDetector":
        from .video_detector import VideoDetector
        return VideoDetector
    if name == "FusionClassifier":
        from .fusion import FusionClassifier
        return FusionClassifier
    raise AttributeError(f"module 'detection' has no attribute {name!r}")
