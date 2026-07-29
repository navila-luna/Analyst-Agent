import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.qa import ask


def format_answer(answer) -> str:
    if not answer.references:
        return answer.text
    ref_lines = [
        f"[{r.index}] {r.doc_name} (chunk {r.chunk_index}, similarity={r.similarity})"
        for r in answer.references
    ]
    return f"{answer.text}\n\n## References\n" + "\n".join(ref_lines)


if __name__ == "__main__":
    print("Ask a question about the sample docs (Ctrl+C to quit):")
    while True:
        question = input("> ")
        print(format_answer(ask(question)))
        print()
