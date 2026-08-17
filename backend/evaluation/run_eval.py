import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.index import query
from app.qa import ask

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def run_eval() -> list[dict]:
    test_cases = json.loads(TEST_CASES_PATH.read_text())
    results = []

    for case in test_cases:
        question = case["question"]

        # Raw retrieval distance, independent of the guardrail - needed later
        # to tune DISTANCE_THRESHOLD against real data instead of guessing.
        retrieval = query(question, top_k=5)
        distances = retrieval["distances"][0]
        best_distance = min(distances) if distances else None

        answer = ask(question)

        results.append(
            {
                **case,
                "actual_answer": answer.text,
                "actual_in_docs": answer.in_docs,
                "actual_references": [
                    {"doc_name": r.doc_name, "chunk_index": r.chunk_index, "similarity": r.similarity}
                    for r in answer.references
                ],
                "best_distance": best_distance,
            }
        )

    return results


if __name__ == "__main__":
    results = run_eval()
    print(f"Ran {len(results)} test cases\n")
    for r in results:
        correct_classification = r["answerable"] == r["actual_in_docs"]
        status = "OK  " if correct_classification else "MISS"
        dist = f"{r['best_distance']:.3f}" if r["best_distance"] is not None else "  n/a"
        print(f"[{status}] dist={dist}  {r['question'][:60]}")
