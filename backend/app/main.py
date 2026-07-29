from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.index import index_folder
from app.qa import ask

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
