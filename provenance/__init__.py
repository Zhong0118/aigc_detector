"""provenance 包的统一导出入口。

provenance 层负责“溯源证据”，它和 detection 层不是一回事：
- detection 判断内容像不像 AI
- provenance 尝试说明内容来源、凭证、水印、历史匹配、可能生成模型

当前真实状态：
- `pipeline.py` 已经把溯源流程串起来
- `c2pa_reader.py` 是轻量占位 parser，不是完整 c2patool
- `watermark_decoder.py` 是图片 LSB 占位，不是 Meta Seal
- `fingerprint_registry.py` 是简单数据库 exact match
- `attribution.py` 是本地 MLP scaffold，未训练时不能作为真实归因

后续重点：
- 接官方 c2patool
- 接 Meta Seal / AudioSeal / TextSeal / VideoSeal
- 建已知内容指纹库
- 训练或替换真实 model attribution 模型
"""

def __getattr__(name):
    """按需导出 provenance 层常用工具。

    使用懒加载可以避免一导入 provenance 包就加载 torch 或其他重依赖。
    """
    if name == "read_c2pa_metadata":
        # C2PA metadata 读取函数，当前是轻量占位 parser。
        from .c2pa_reader import read_c2pa_metadata
        return read_c2pa_metadata
    if name == "decode_watermark":
        # 图片水印占位 decoder，未来可替换/扩展为 Meta Seal family。
        from .watermark_decoder import decode_watermark
        return decode_watermark
    if name == "FingerprintRegistry":
        # 指纹库 exact match 工具。
        from .fingerprint_registry import FingerprintRegistry
        return FingerprintRegistry
    if name == "ModelAttribution":
        # 本地归因模型 scaffold。
        from .attribution import ModelAttribution
        return ModelAttribution
    raise AttributeError(f"module 'provenance' has no attribute {name!r}")
