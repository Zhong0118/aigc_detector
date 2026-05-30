"""指纹库查询与注册。

FingerprintRegistry 用来把当前内容 fingerprint 和历史内容/已知内容库做匹配。
它属于 provenance 层，因为命中历史样本或已知模型样本可以作为来源证据。

当前实现：
- exact match：文本、音频、视频、未知文件都先做完全一致匹配
- image phash near match：图片感知哈希支持汉明距离近似匹配
- 复用 storage.contents 表，不额外引入新数据库

后续建议：
- audio/video embedding 相似度匹配
- 已知 AI 样本 registry 导入
- Elasticsearch / FAISS / pgvector 等相似检索后端
"""

from datetime import datetime
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
    match_type: str = "exact"
    distance: Optional[int] = None
    modality: Optional[str] = None
    filename: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """转换成 API/报告可序列化字典。"""
        return {
            "content_id": self.content_id,
            "fingerprint": self.fingerprint,
            "similarity": round(self.similarity, 4),
            "source_model": self.source_model,
            "match_type": self.match_type,
            "distance": self.distance,
            "modality": self.modality,
            "filename": self.filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FingerprintRegistry:
    """基于数据库的简单指纹库。"""

    def lookup(
        self,
        fingerprint: str,
        modality: str | None = None,
        max_distance: int = 8,
        limit: int = 5,
        exclude_content_id: int | None = None,
    ) -> List[RegistryMatch]:
        """查找历史内容。

        - 所有模态先做 fingerprint 完全一致匹配
        - image 且 fingerprint 看起来像 pHash 时，再做汉明距离近似匹配
        """
        session = get_session()
        try:
            query = session.query(Content)
            if modality:
                query = query.filter(Content.modality == modality)
            if exclude_content_id is not None:
                query = query.filter(Content.id != exclude_content_id)

            candidates = query.order_by(Content.created_at.desc()).all()
            matches: list[RegistryMatch] = []

            for content in candidates:
                if not content.fingerprint:
                    continue
                if content.fingerprint == fingerprint:
                    matches.append(self._match_from_content(content, 1.0, "exact", 0))
                elif modality == "image":
                    distance = self._hex_hamming_distance(fingerprint, content.fingerprint)
                    if distance is not None and distance <= max_distance:
                        bit_count = max(len(fingerprint), len(content.fingerprint)) * 4
                        similarity = 1.0 - (distance / bit_count)
                        matches.append(
                            self._match_from_content(content, similarity, "near_phash", distance)
                        )

            matches.sort(key=lambda match: (match.similarity, -(match.distance or 0)), reverse=True)
            return matches[:limit]
        finally:
            session.close()

    def _match_from_content(
        self,
        content: Content,
        similarity: float,
        match_type: str,
        distance: int | None,
    ) -> RegistryMatch:
        """从 Content ORM 对象构造 RegistryMatch。"""
        return RegistryMatch(
            content_id=content.id,
            fingerprint=content.fingerprint,
            similarity=similarity,
            source_model=content.source_model,
            match_type=match_type,
            distance=distance,
            modality=content.modality,
            filename=content.filename,
            created_at=content.created_at,
        )

    def _hex_hamming_distance(self, left: str, right: str) -> int | None:
        """计算两个十六进制 pHash 字符串的汉明距离。"""
        try:
            if len(left) != len(right):
                return None
            left_value = int(left, 16)
            right_value = int(right, 16)
            return (left_value ^ right_value).bit_count()
        except ValueError:
            return None

    def lookup_as_dicts(
        self,
        fingerprint: str,
        modality: str | None = None,
        max_distance: int = 8,
        limit: int = 5,
        exclude_content_id: int | None = None,
    ) -> list[dict]:
        """lookup 的字典版，方便 pipeline 直接放进 JSON。"""
        return [
            match.to_dict()
            for match in self.lookup(
                fingerprint,
                modality=modality,
                max_distance=max_distance,
                limit=limit,
                exclude_content_id=exclude_content_id,
            )
        ]

    def register(
        self,
        fingerprint: str,
        source_model: Optional[str] = None,
        filename: str | None = None,
        modality: str | None = None,
        file_hash: str | None = None,
    ) -> int:
        """注册一个 fingerprint 到内容表。

        source_model 可选，用于标记该内容来自哪个已知模型。
        """
        session = get_session()
        try:
            content = Content(
                filename=filename,
                modality=modality,
                fingerprint=fingerprint,
                file_hash=file_hash or fingerprint,
                source_model=source_model,
            )
            session.add(content)
            session.commit()
            return int(content.id)
        finally:
            session.close()
