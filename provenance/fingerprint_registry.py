from typing import Optional, List
from dataclasses import dataclass

from storage.database import get_session
from storage.models import Content


@dataclass
class RegistryMatch:
    content_id: int
    fingerprint: str
    similarity: float
    source_model: Optional[str]


class FingerprintRegistry:
    def lookup(self, fingerprint: str) -> List[RegistryMatch]:
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
        session = get_session()
        try:
            content = Content(fingerprint=fingerprint, source_model=source_model)
            session.add(content)
            session.commit()
            return content.id
        finally:
            session.close()
