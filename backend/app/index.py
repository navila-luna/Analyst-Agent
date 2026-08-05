from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.ingest import build_chunks
from app.manifest import Manifest, build_chunk_hashes, build_file_hashes, diff_hashes, load_manifest, save_manifest

CHROMA_DIR = Path(__file__).parent.parent / "chroma_data"
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_folder(folder: Path) -> dict:
    """Incrementally (re)index folder using two tiers of change detection:

    1. Whole-file hash: cheap pre-check that skips reading/chunking a file
       entirely if its content hasn't changed at all (matters for large files).
    2. Per-chunk hash: for files that did change, only the specific chunks
       whose content actually differs get re-embedded - not the whole file.
    """
    manifest = load_manifest()
    current_file_hashes = build_file_hashes(folder)
    file_diff = diff_hashes(manifest.files, current_file_hashes)

    collection = get_collection()

    # Deleted files: we never re-read them, so we don't know their exact chunk
    # ids - fall back to a metadata filter to purge all their chunks at once.
    if file_diff.deleted:
        collection.delete(where={"doc_name": {"$in": file_diff.deleted}})

    files_to_process = set(file_diff.new) | set(file_diff.changed)
    new_chunk_manifest = dict(manifest.chunks)
    chunks_embedded = 0

    if files_to_process:
        chunks = build_chunks(folder, doc_names=files_to_process)
        current_chunk_hashes = build_chunk_hashes(chunks)

        # Only diff against previous chunk hashes for these same files, so
        # untouched files' chunks aren't mistaken for "deleted".
        #
        # Scalability: every chunk hash lives in one dict under a key like
        # "file.md::3" - there's no way to grab "just this file's chunks"
        # without scanning all of them. So this scan (and the two below) cost
        # O(total corpus), not O(files_to_process). A dict per file would fix
        # that; revisit at Phase 7 (scale/deployment) if it becomes real
        # latency.
        old_chunk_hashes = {
            cid: h for cid, h in manifest.chunks.items()
            if cid.split("::", 1)[0] in files_to_process
        }
        chunk_diff = diff_hashes(old_chunk_hashes, current_chunk_hashes)

        ids_to_remove = chunk_diff.changed + chunk_diff.deleted
        if ids_to_remove:
            collection.delete(ids=ids_to_remove)

        ids_to_add = set(chunk_diff.new) | set(chunk_diff.changed)
        chunks_to_embed = [c for c in chunks if f"{c.doc_name}::{c.chunk_index}" in ids_to_add]

        if chunks_to_embed:
            model = get_model()
            ids = [f"{c.doc_name}::{c.chunk_index}" for c in chunks_to_embed]
            texts = [c.text for c in chunks_to_embed]
            metadatas = [{"doc_name": c.doc_name, "chunk_index": c.chunk_index} for c in chunks_to_embed]
            embeddings = model.encode(texts).tolist()
            collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        chunks_embedded = len(chunks_to_embed)

        # Same flat-key scan cost as old_chunk_hashes above.
        new_chunk_manifest = {
            cid: h for cid, h in new_chunk_manifest.items()
            if cid.split("::", 1)[0] not in files_to_process
        }
        new_chunk_manifest.update(current_chunk_hashes)

    if file_diff.deleted:
        # Same flat-key scan cost as old_chunk_hashes above.
        deleted_set = set(file_diff.deleted)
        new_chunk_manifest = {
            cid: h for cid, h in new_chunk_manifest.items()
            if cid.split("::", 1)[0] not in deleted_set
        }

    save_manifest(Manifest(files=current_file_hashes, chunks=new_chunk_manifest))

    return {
        "files_new": file_diff.new,
        "files_changed": file_diff.changed,
        "files_unchanged": file_diff.unchanged,
        "files_deleted": file_diff.deleted,
        "chunks_embedded": chunks_embedded,
    }


def query(question: str, top_k: int = 5):
    """Embed a question and return the top_k most similar chunks from the collection."""
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results
