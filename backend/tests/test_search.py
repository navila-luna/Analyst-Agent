import sys
import types
import unittest
from unittest.mock import patch

fake_index = types.ModuleType("app.index")
fake_index.query = None

with patch.dict(sys.modules, {"app.index": fake_index}):
    import app.search as search_module

search_documents = search_module.search_documents


class SearchDocumentsTests(unittest.TestCase):
    @patch.object(search_module, "query")
    def test_returns_chunks_with_metadata_and_similarity(self, mock_query):
        mock_query.return_value = {
            "documents": [["First chunk", "Second chunk"]],
            "metadatas": [[
                {"doc_name": "guide.md", "chunk_index": 0},
                {"doc_name": "guide.md", "chunk_index": 1},
            ]],
            "distances": [[0.123, 0.456]],
        }

        results = search_documents("How does it work?")

        mock_query.assert_called_once_with("How does it work?", top_k=5)
        self.assertEqual(results, [
            {
                "text": "First chunk",
                "doc_name": "guide.md",
                "chunk_index": 0,
                "similarity": 0.88,
            },
            {
                "text": "Second chunk",
                "doc_name": "guide.md",
                "chunk_index": 1,
                "similarity": 0.54,
            },
        ])

    @patch.object(search_module, "query")
    def test_returns_empty_list_when_no_chunks_are_found(self, mock_query):
        mock_query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        self.assertEqual(search_documents("Missing topic"), [])


if __name__ == "__main__":
    unittest.main()
