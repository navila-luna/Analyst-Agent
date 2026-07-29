from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import models  # noqa: F401 - registers models with Base.metadata
from app.config_store import get_or_create_config
from app.db import Base, engine, get_db
from app.index import index_folder
from app.qa import ask

Base.metadata.create_all(bind=engine)

UPLOAD_DIR = Path(__file__).parent.parent / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Knowledge Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class ConfigPayload(BaseModel):
    tone: str
    answer_format: str
    require_citations: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    answer = ask(request.question, top_k=request.top_k)
    return {
        "answer": answer.text,
        "in_docs": answer.in_docs,
        "references": [asdict(r) for r in answer.references],
    }


@app.post("/index")
async def index_endpoint(files: list[UploadFile]):
    for file in files:
        destination = UPLOAD_DIR / file.filename
        destination.write_bytes(await file.read())

    result = index_folder(UPLOAD_DIR)
    return result


@app.get("/config")
def get_config(db: Session = Depends(get_db)):
    config = get_or_create_config(db)
    return {
        "tone": config.tone,
        "answer_format": config.answer_format,
        "require_citations": config.require_citations,
    }


@app.put("/config")
def update_config(payload: ConfigPayload, db: Session = Depends(get_db)):
    config = get_or_create_config(db)
    config.tone = payload.tone
    config.answer_format = payload.answer_format
    config.require_citations = payload.require_citations
    db.commit()
    db.refresh(config)
    return {
        "tone": config.tone,
        "answer_format": config.answer_format,
        "require_citations": config.require_citations,
    }
