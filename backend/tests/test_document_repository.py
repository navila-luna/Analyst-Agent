import tempfile
import unittest
from pathlib import Path

from app.document_repository import (
    MAX_CHUNKS_PER_READ,
    DocumentNotFoundError,
    FilesystemDocumentRepository,
    InvalidDocumentNameError,
)


class DocumentRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repository = FilesystemDocumentRepository(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_lists_supported_documents_in_name_order(self):
        (self.root / "b.txt").write_text("B", encoding="utf-8")
        (self.root / "a.md").write_text("A", encoding="utf-8")
        (self.root / "ignored.pdf").write_bytes(b"pdf")

        documents = self.repository.list_documents()

        self.assertEqual([item["doc_name"] for item in documents], ["a.md", "b.txt"])
        self.assertEqual([item["chunk_count"] for item in documents], [1, 1])

    def test_reads_ordered_chunks_with_pagination(self):
        (self.root / "long.txt").write_text("A" * 1800, encoding="utf-8")

        first = self.repository.read_document("long.txt", 0, 2)
        second = self.repository.read_document("long.txt", first["next_chunk"], 2)

        self.assertEqual([item["chunk_index"] for item in first["chunks"]], [0, 1])
        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_chunk"], 2)
        self.assertEqual([item["chunk_index"] for item in second["chunks"]], [2])
        self.assertFalse(second["has_more"])
        self.assertIsNone(second["next_chunk"])

    def test_clamps_requested_chunk_count(self):
        (self.root / "long.txt").write_text("A" * 9000, encoding="utf-8")

        result = self.repository.read_document("long.txt", 0, 500)

        self.assertEqual(result["max_chunks_applied"], MAX_CHUNKS_PER_READ)
        self.assertLessEqual(result["chunks_returned"], MAX_CHUNKS_PER_READ)

    def test_rejects_path_traversal(self):
        with self.assertRaises(InvalidDocumentNameError):
            self.repository.read_document("../.env")

    def test_rejects_unsupported_extension(self):
        (self.root / "notes.pdf").write_bytes(b"pdf")
        with self.assertRaises(InvalidDocumentNameError):
            self.repository.read_document("notes.pdf")

    def test_reports_missing_document(self):
        with self.assertRaises(DocumentNotFoundError):
            self.repository.read_document("missing.txt")


if __name__ == "__main__":
    unittest.main()
