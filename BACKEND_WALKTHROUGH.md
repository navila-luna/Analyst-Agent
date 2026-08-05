# Reading the Backend: A Mental Model for the RAG Pipeline

This is not a line-by-line code tour. It's a guide to *how to approach* this
codebase if your goal is to actually understand it — what order to read files
in, what questions to ask at each one, and how to judge whether a design
decision is working or needs to change.

**Scope:** the backend's retrieval pipeline (`backend/app/`), which is where
this project's actual "RAG thinking" lives. The frontend (`frontend/`) is the
chat UI on top of it — worth understanding eventually, but it's presentation,
not retrieval logic. If your goal is "how does this bot turn a question into
a grounded answer," you never need to open `frontend/` to get there.

New to RAG itself? [RAG_PIPELINE.md](RAG_PIPELINE.md) covers what the
retrieve/augment/generate loop actually is and maps each backend file to the
part it plays, with a runtime walkthrough of both the upload flow and the
question-asking flow.

Once you can follow *what* the code does, see
[THINKING_LIKE_AN_ENGINEER.md](THINKING_LIKE_AN_ENGINEER.md) for how to form
an opinion on *whether it's a good way to do it* — scalability, tradeoffs,
and when to question a design instead of assuming it's settled.

## Step 0: Decide what you're trying to answer before you read anything

"Understand the codebase" is too big a goal to read toward — you'll skim
everything and retain nothing. Pick one concrete question per sitting, e.g.:

- "How does a question turn into an answer?"
- "How does the bot decide it doesn't know something, instead of making
  something up?"
- "How do uploaded documents get turned into something searchable?"
- "What happens if I upload the same file twice, or edit one file slightly?"

Each question below maps to one or two files. Read only what answers your
current question — you can come back for the rest later.

## The one-sentence model of RAG

**Retrieval-Augmented Generation = look up the most relevant snippets of your
own documents first, then hand those snippets to the LLM as "open-book notes"
so it answers from your docs instead of guessing from what it memorized
during training.**

Every file in this pipeline exists to serve one of two halves of that
sentence: *retrieval* (finding the right snippets) or *generation*
(turning snippets + a question into an answer).

## Reading order (follow the data, not the file tree)

Read these in the order data actually flows through the system — not
alphabetically, not in the order they appear in the folder.

**1. [ingest.py](backend/app/ingest.py) — turning raw files into bite-sized chunks**
Answers: "how do uploaded documents get prepared?"
Look at `CHUNK_SIZE` and `CHUNK_OVERLAP` — documents get cut into overlapping
~800-character pieces because embedding models and LLM context windows both
work on small pieces, not entire files. The overlap exists so an idea that
happens to fall right on a chunk boundary isn't cut in half and lost.

**2. [manifest.py](backend/app/manifest.py) — where hashing files is applied**
Answers: "what happens if I re-upload a folder where only one file changed?"
This is bookkeeping, not core RAG logic: it hashes files and individual
chunks so re-indexing only touches what actually changed. 
Engineering Perspective: understanding what the methods do and why 
certain data structures were used to solve certain problems.

**3. [index.py](backend/app/index.py) — turning chunks into searchable vectors**
Answers: "how does the bot 'search' documents instead of just keyword-matching?"
This is where chunks get embedded (turned into number vectors via
`all-MiniLM-L6-v2`) and stored in ChromaDB. The `query()` function at the
bottom is the one that matters most for a first read: given a question, it
embeds it the same way and asks ChromaDB for the most similar chunks by
cosine distance. `index_folder()` is the incremental-indexing machinery built
on top of `manifest.py` — worth a second pass, not your first.

**4. [qa.py](backend/app/qa.py) — the actual RAG loop: question in, answer out**
Answers: "how does a question become an answer, and how does the bot know
when to say 'I don't know'?"
This is the most important file in the pipeline. Read `ask()` top to bottom:
it calls `query()` from `index.py`, checks the retrieved chunks' distance
against `DISTANCE_THRESHOLD` (the guardrail that makes the bot refuse to
answer instead of hallucinating from weak matches), builds a prompt via
`build_prompt()` and `build_system_prompt()`, and calls the OpenAI API.

**5. [main.py](backend/app/main.py) — where the pipeline becomes an API**
Answers: "how does the frontend actually talk to this?"
Thin FastAPI wrapper: `/ask` calls `qa.ask()`, `/index` calls
`index.index_folder()`, `/config` reads/writes the tone and formatting
settings that `qa.py` uses to build its system prompt. This is a good place
to stop — everything past this file is frontend/UI concerns, not retrieval
logic.

## How to evaluate a RAG decision — and build on the previous one

Every choice in this pipeline is a tunable knob, not a fixed fact:

- Chunk size and overlap ([ingest.py](backend/app/ingest.py))
- Embedding model ([index.py](backend/app/index.py) — `all-MiniLM-L6-v2`)
- The "give up and say I don't know" distance threshold ([qa.py](backend/app/qa.py))
- The generation model (`gpt-4o-mini` in [qa.py](backend/app/qa.py))
- The system prompt / tone / citation requirements ([qa.py](backend/app/qa.py))

None of these were derived from a formula — they were picked, tried, and (in
one documented case) adjusted based on what was observed. Look at the comment
above `DISTANCE_THRESHOLD` in [qa.py](backend/app/qa.py:14-19): it explains
*why* the value is 0.75 and not 0.6 — a real answer scored 0.56-0.62 in
testing, so 0.6 left almost no margin. That comment is doing exactly what
"evaluating and building on a previous approach" means in practice: someone
tried a value, watched it fail on real input, changed one variable, and wrote
down why — so the next person (including future-you) doesn't have to
re-discover it from scratch or guess whether the current number is
load-bearing.

Until the formal eval pipeline exists (see "Phase 6" in
[PLAN.md](PLAN.md) — ground-truth QA generation, LLM-as-judge, cosine
similarity, precision/recall), evaluating a change here is manual and
qualitative:

1. Ask a handful of real questions you know the answer to from the docs.
2. Check: did it answer correctly, and does it cite the right source chunks?
3. Ask something *not* in the docs — does it correctly refuse instead of
   guessing?
4. If something's off, change **one** knob at a time (not three at once —
   you won't know which one fixed or broke it) and repeat.
5. Write down what you changed and why, the way the `DISTANCE_THRESHOLD`
   comment does — that note is what lets the *next* iteration start from
   "here's what we already learned" instead of from zero.

## What to skip for now

- `frontend/` entirely, if your goal is understanding retrieval/generation.
- [config_store.py](backend/app/config_store.py) and [models.py](backend/app/models.py) — small,
  supporting pieces for the per-team tone/format settings; only worth reading
  once you already understand `qa.py` and want to know where `config.tone`
  comes from.
