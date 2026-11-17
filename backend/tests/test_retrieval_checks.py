import unittest

from evaluation.retrieval_checks import RetrievalCase, assess
"""
    Note to self. The script does not call OpenAI. 
    It evaluates the existing local embedding model and ChromaDB collection
    The statuses mean the following:
            - PASS: retrieval met the declared expectation.
            - FLAG: expected evidence was found, but its score or ranking is concerning.
            - MISS: an expected chunk was absent from the top five.
            - REVIEW: the question is ambiguous and requires Nicole's judgment.
""" 

def result(doc_name, chunk_index, similarity):
    return {
        "doc_name": doc_name,
        "chunk_index": chunk_index,
        "similarity": similarity,
        "text": "unused",
    }


class RetrievalChecksTests(unittest.TestCase):
    def test_answerable_case_passes_when_expected_chunk_is_found(self):
        case = RetrievalCase("direct", "question", ("guide.md::1",))

        status, _ = assess(case, [result("guide.md", 1, 0.6)], 0.25)

        self.assertEqual(status, "PASS")

    def test_answerable_case_flags_a_low_scoring_match(self):
        case = RetrievalCase("direct", "question", ("guide.md::1",))

        status, _ = assess(case, [result("guide.md", 1, 0.2)], 0.25)

        self.assertEqual(status, "FLAG")

    def test_single_expected_chunk_is_flagged_when_not_ranked_first(self):
        case = RetrievalCase("direct", "question", ("guide.md::1",))

        status, _ = assess(
            case,
            [
                result("other.md", 0, 0.7),
                result("guide.md", 1, 0.6),
            ],
            0.25,
        )

        self.assertEqual(status, "FLAG")

    def test_expected_score_is_checked_instead_of_unrelated_top_score(self):
        case = RetrievalCase("direct", "question", ("guide.md::1",))

        status, _ = assess(
            case,
            [
                result("other.md", 0, 0.7),
                result("guide.md", 1, 0.2),
            ],
            0.25,
        )

        self.assertEqual(status, "FLAG")

    def test_answerable_case_reports_missing_expected_chunk(self):
        case = RetrievalCase("direct", "question", ("guide.md::1",))

        status, _ = assess(case, [result("other.md", 0, 0.6)], 0.25)

        self.assertEqual(status, "MISS")

    def test_unsupported_case_passes_below_threshold(self):
        case = RetrievalCase("unsupported", "question", answerable=False)

        status, _ = assess(case, [result("guide.md", 0, 0.06)], 0.25)

        self.assertEqual(status, "PASS")

    def test_unsupported_case_is_flagged_at_threshold(self):
        case = RetrievalCase("unsupported", "question", answerable=False)

        status, _ = assess(case, [result("guide.md", 0, 0.25)], 0.25)

        self.assertEqual(status, "FLAG")

    def test_ambiguous_case_requires_review(self):
        case = RetrievalCase("ambiguous", "question", answerable=None)

        status, _ = assess(case, [result("guide.md", 0, 0.5)], 0.25)

        self.assertEqual(status, "REVIEW")


if __name__ == "__main__":
    unittest.main()
