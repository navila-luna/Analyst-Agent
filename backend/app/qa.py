from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from app.index import query

load_dotenv()
client = OpenAI()

# Cosine distance: 0 = identical, 2 = opposite. Anything worse than this is
# treated as "not actually in the docs" and short-circuits before the LLM call.
DISTANCE_THRESHOLD = 0.6

NOT_IN_DOCS_MESSAGE = "I don't know — the provided documents don't cover this."

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context. Cite sources inline using [n] matching the numbered context blocks. "
    "If the context doesn't contain the answer, say you don't know."
)


@dataclass
class Reference:
    index: int
    doc_name: str
    chunk_index: int
    similarity: float


@dataclass
class Answer:
    text: str
    references: list[Reference]
    in_docs: bool


def build_prompt(question: str, chunks: list[str]) -> str:
    numbered_context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
    return f"Context:\n{numbered_context}\n\nQuestion: {question}"


def ask(question: str, top_k: int = 5) -> Answer:
    results = query(question, top_k=top_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not distances or min(distances) > DISTANCE_THRESHOLD:
        return Answer(text=NOT_IN_DOCS_MESSAGE, references=[], in_docs=False)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    answer_text = response.choices[0].message.content

    references = [
        Reference(
            index=i + 1,
            doc_name=meta["doc_name"],
            chunk_index=meta["chunk_index"],
            similarity=round(1 - dist, 2),
        )
        for i, (meta, dist) in enumerate(zip(metadatas, distances))
    ]

    return Answer(text=answer_text, references=references, in_docs=True)
