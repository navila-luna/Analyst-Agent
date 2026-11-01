from pathlib import Path

from app.ingest import SUPPORTED_EXTENSIONS, chunk_text


DEFAULT_CHUNKS_PER_READ = 5
MAX_CHUNKS_PER_READ = 10
UPLOAD_DIR = Path(__file__).parent.parent / "uploaded_docs"


class DocumentNotFoundError(ValueError):
    pass


class InvalidDocumentNameError(ValueError):
    pass


class FilesystemDocumentRepository:
    def __init__(self, root: Path = UPLOAD_DIR):
        self.root = root.resolve()

    def list_documents(self) -> list[dict]:
        documents = []
        if not self.root.exists():
            return documents

        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = path.read_text(encoding="utf-8")
            documents.append({
                "doc_name": path.name,
                "size_bytes": path.stat().st_size,
                "chunk_count": len(chunk_text(text)),
            })
        return documents

    def read_document(
        self,
        doc_name: str,
        start_chunk: int = 0,
        max_chunks: int = DEFAULT_CHUNKS_PER_READ,
    ) -> dict:
        if start_chunk < 0:
            raise ValueError("start_chunk must be zero or greater")

        requested_limit = max(1, max_chunks)
        actual_limit = min(requested_limit, MAX_CHUNKS_PER_READ)
        path = self._resolve_document(doc_name)
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        selected = chunks[start_chunk:start_chunk + actual_limit]
        next_chunk = start_chunk + len(selected)

        return {
            "doc_name": path.name,
            "total_chunks": len(chunks),
            "start_chunk": start_chunk,
            "chunks_returned": len(selected),
            "next_chunk": next_chunk if next_chunk < len(chunks) else None,
            "has_more": next_chunk < len(chunks),
            "max_chunks_applied": actual_limit,
            "chunks": [
                {"chunk_index": start_chunk + index, "text": text}
                for index, text in enumerate(selected)
            ],
        }

    def _resolve_document(self, doc_name: str) -> Path:
        if not doc_name or Path(doc_name).name != doc_name:
            raise InvalidDocumentNameError("doc_name must be a filename, not a path")

        candidate = (self.root / doc_name).resolve()
        if candidate.parent != self.root:
            raise InvalidDocumentNameError("document must be inside uploaded_docs")
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidDocumentNameError("unsupported document type")
        if not candidate.is_file():
            raise DocumentNotFoundError(f"document not found: {doc_name}")
        return candidate


repository = FilesystemDocumentRepository()


def list_documents() -> list[dict]:
    """List readable documents in the private knowledge base."""
    return repository.list_documents()


def read_document(
    doc_name: str,
    start_chunk: int = 0,
    max_chunks: int = DEFAULT_CHUNKS_PER_READ,
) -> dict:
    """Read an ordered, paginated set of chunks from one document."""
    return repository.read_document(doc_name, start_chunk, max_chunks)
