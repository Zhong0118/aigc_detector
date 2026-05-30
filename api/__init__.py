"""API 包。

这个文件本身不写业务逻辑，只负责把 `api/` 标记成一个 Python 包。
有了它以后，其他文件才能使用 `from api.routes import router`
或者 `from api.analysis import get_analysis_service` 这类包内导入。
"""
