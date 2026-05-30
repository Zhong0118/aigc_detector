"""指纹库查询与注册。

FingerprintRegistry 用来把当前内容 fingerprint 和历史内容/已知内容库做匹配。
它属于 provenance 层，因为命中历史样本或已知模型样本可以作为来源证据。

当前实现：
- 只做数据库 exact match
- 不做近似匹配
- 不做向量检索

后续建议：
- image phash 汉明距离匹配
- audio/video embedding 相似度匹配
- 已知 AI 样本 registry 导入
- Elasticsearch / FAISS / pgvector 等相似检索后端
"""

from typing import Optional, List
from dataclasses import dataclass

from storage.database import get_session
from storage.models import Content


@dataclass
class RegistryMatch:
    """一次指纹库命中结果。"""

    content_id: int
    fingerprint: str
    similarity: float
    source_model: Optional[str]


class FingerprintRegistry:
    """基于数据库的简单指纹库。"""

    def lookup(self, fingerprint: str) -> List[RegistryMatch]:
        """查找完全相同 fingerprint 的历史内容。"""
        session = get_session()
        try:
            exact = session.query(Content).filter(Content.fingerprint == fingerprint).all()
            return [
                RegistryMatch(
                    content_id=c.id,
                    fingerprint=c.fingerprint,
                    similarity=1.0,
                    source_model=c.source_model,
                )
                for c in exact
            ]
        finally:
            session.close()

    def register(self, fingerprint: str, source_model: Optional[str] = None) -> int:
        """注册一个 fingerprint 到内容表。

        source_model 可选，用于标记该内容来自哪个已知模型。
        """
        session = get_session()
        try:
            content = Content(fingerprint=fingerprint, source_model=source_model)
            session.add(content)
            session.commit()
            return content.id
        finally:
            session.close()
