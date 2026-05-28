import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from storage.database import init_db
from api.routes import router

app = FastAPI(
    title="AIGC Detection & Attribution API",
    version="0.1.0",
    description="Multi-modal AI content detection and provenance analysis",
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}
