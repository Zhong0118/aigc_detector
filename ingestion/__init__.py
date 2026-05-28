def __getattr__(name):
    if name == "load_content":
        from .loader import load_content
        return load_content
    if name == "extract_metadata":
        from .metadata import extract_metadata
        return extract_metadata
    if name == "compute_fingerprint":
        from .fingerprint import compute_fingerprint
        return compute_fingerprint
    raise AttributeError(f"module 'ingestion' has no attribute {name!r}")
