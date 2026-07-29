import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.ingest import Chunk, SUPPORTED_EXTENSIONS

MANIFEST_PATH = Path(__file__).parent.parent / "chroma_data" / "manifest.json"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def build_file_hashes(folder: Path) -> dict[str, str]:
    """doc_name -> whole-file content hash. Cheap pre-check: lets us skip
    reading/chunking a file entirely if it hasn't changed at all."""
    return {
        path.name: hash_bytes(path.read_bytes())
        for path in sorted(folder.iterdir())
        if path.suffix in SUPPORTED_EXTENSIONS
    }


def build_chunk_hashes(chunks: list[Chunk]) -> dict[str, str]:
    """chunk_id ('doc_name::chunk_index') -> content hash of that one chunk."""
    return {f"{c.doc_name}::{c.chunk_index}": hash_text(c.text) for c in chunks}


@dataclass
class Manifest:
    files: dict[str, str]
    chunks: dict[str, str]


def load_manifest() -> Manifest:
    if not MANIFEST_PATH.exists():
        return Manifest(files={}, chunks={})
    data = json.loads(MANIFEST_PATH.read_text())
    return Manifest(files=data.get("files", {}), chunks=data.get("chunks", {}))


def save_manifest(manifest: Manifest) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps({"files": manifest.files, "chunks": manifest.chunks}, indent=2)
    )


@dataclass
class Diff:
    new: list[str]
    changed: list[str]
    unchanged: list[str]
    deleted: list[str]


def diff_hashes(old: dict[str, str], current: dict[str, str]) -> Diff:
    """Classify every key as new, changed, unchanged, or deleted.

    Generic over what the keys represent - used for both doc_name -> file hash
    and chunk_id -> chunk hash comparisons.
    """
    new, changed, unchanged = [], [], []

    for key, current_hash in current.items():
        if key not in old:
            new.append(key)
        elif old[key] != current_hash:
            changed.append(key)
        else:
            unchanged.append(key)

    deleted = [key for key in old if key not in current]

    return Diff(new=new, changed=changed, unchanged=unchanged, deleted=deleted)
