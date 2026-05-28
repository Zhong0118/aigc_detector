def __getattr__(name):
    if name == "read_c2pa_metadata":
        from .c2pa_reader import read_c2pa_metadata
        return read_c2pa_metadata
    if name == "decode_watermark":
        from .watermark_decoder import decode_watermark
        return decode_watermark
    if name == "FingerprintRegistry":
        from .fingerprint_registry import FingerprintRegistry
        return FingerprintRegistry
    if name == "ModelAttribution":
        from .attribution import ModelAttribution
        return ModelAttribution
    raise AttributeError(f"module 'provenance' has no attribute {name!r}")
