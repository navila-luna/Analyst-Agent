import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from app.ingest import load_documents

load_dotenv()
client = OpenAI()

DOCS_FOLDER = Path(__file__).parent.parent / "uploaded_docs"
OUTPUT_PATH = Path(__file__).parent / "test_cases.json"

QUESTIONS_PER_DOC = 2

# Deliberately out-of-scope questions - none of these should be answerable
# from the indexed docs, regardless of what gets uploaded.
UNANSWERABLE_QUESTIONS = [
    "What's the capital of France?",
    "What's our company's parental leave policy?",
    "How do I set up a VPN connection?",
    "What's the weather forecast for tomorrow?",
    "Who is the CEO of our biggest competitor?",
    "When will the conference be in France?",
]


def generate_questions_for_doc(doc_name: str, text: str) -> list[dict]:
    prompt = (
        f"Given this internal document, generate {QUESTIONS_PER_DOC} question/answer "
        "pairs a team member might realistically ask, where the answer is directly "
        "and clearly stated in the document. Return a JSON object of the form "
        '{"questions": [{"question": "...", "expected_answer": "..."}]}.\n\n'
        f"Document ({doc_name}):\n{text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    parsed = json.loads(response.choices[0].message.content)
    return [
        {
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "expected_doc": doc_name,
            "answerable": True,
        }
        for item in parsed["questions"]
    ]


def build_test_cases() -> list[dict]:
    test_cases = []

    for doc_name, text in load_documents(DOCS_FOLDER).items():
        test_cases.extend(generate_questions_for_doc(doc_name, text))

    for question in UNANSWERABLE_QUESTIONS:
        test_cases.append(
            {
                "question": question,
                "expected_answer": None,
                "expected_doc": None,
                "answerable": False,
            }
        )

    return test_cases


if __name__ == "__main__":
    cases = build_test_cases()
    OUTPUT_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Generated {len(cases)} test cases -> {OUTPUT_PATH}")
