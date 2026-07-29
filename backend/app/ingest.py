from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt"}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    doc_name: str
    chunk_index: int
    text: str


def load_documents(folder: Path) -> dict[str, str]:
    """Read every supported file in folder into {filename: full_text}."""
    documents = {}
    for path in sorted(folder.iterdir()):
        if path.suffix in SUPPORTED_EXTENSIONS:
            documents[path.name] = path.read_text(encoding="utf-8")
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks, on character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_chunks(folder: Path, doc_names: set[str] | None = None) -> list[Chunk]:
    """Load documents in folder and split each into Chunks.

    If doc_names is given, only those files are loaded/chunked (used for
    incremental re-indexing of just the new/changed files).
    """
    all_chunks = []
    for doc_name, text in load_documents(folder).items():
        if doc_names is not None and doc_name not in doc_names:
            continue
        for i, piece in enumerate(chunk_text(text)):
            all_chunks.append(Chunk(doc_name=doc_name, chunk_index=i, text=piece))
    return all_chunks
