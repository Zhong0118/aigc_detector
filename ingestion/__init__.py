"""ingestion 包的统一导出入口。

ingestion 层负责“输入内容进入系统后的第一步处理”：
- 判断文件属于 text/image/audio/video 哪种模态
- 读取原始内容
- 提取 metadata
- 计算 fingerprint

这里使用 `__getattr__` 做懒加载：只有外部真正访问某个函数时，
才导入对应模块。这样可以减少启动时对 PIL/librosa/cv2 等重依赖的加载。
"""

def __getattr__(name):
    """按需导出 ingestion 层常用函数。

    外部可以直接写：
    `from ingestion import load_content, extract_metadata, compute_fingerprint`

    Python 会在这里根据函数名去对应文件里导入真实函数。
    """
    if name == "load_content":
        # 读取文件并判断模态，真实函数在 ingestion/loader.py。
        from .loader import load_content
        return load_content
    if name == "extract_metadata":
        # 提取文件基础信息和图片 EXIF，真实函数在 ingestion/metadata.py。
        from .metadata import extract_metadata
        return extract_metadata
    if name == "compute_fingerprint":
        # 计算内容指纹，真实函数在 ingestion/fingerprint.py。
        from .fingerprint import compute_fingerprint
        return compute_fingerprint
    raise AttributeError(f"module 'ingestion' has no attribute {name!r}")
