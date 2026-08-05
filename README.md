# Analyst-Agent
An agent that helps answer already asked questions in uploaded files.

A RAG (Retrieval-Augmented Generation) knowledge bot: upload your team's docs,
ask questions in a chat UI, and get answers grounded in those docs with source
citations. See [PLAN.md](PLAN.md) for the full design and phased build plan.

New to this codebase? [BACKEND_WALKTHROUGH.md](BACKEND_WALKTHROUGH.md) is a
guide to how to approach reading the backend/RAG pipeline — what order to
read files in and how to think about evaluating RAG decisions — rather than
a line-by-line tour.

## Prerequisites

- **Python 3.11+**
- **Node.js 20+** and npm (the frontend's Vite version requires 20+; see `.nvmrc`)
- **Homebrew** (macOS) — used to install Postgres
- **PostgreSQL 16** — installed via Homebrew (see setup below)
- An **OpenAI API key** — used for answer generation

## Before you begin: `.env` and `.gitignore`

`.env` holds your personal secrets (your API key, your database address). `.gitignore` tells git to never commit that file, so your secrets never end up on GitHub. That's the whole idea — if you want the deeper why, [GitHub's guide to ignoring files](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files) covers it well.

Setup step 2 below walks you through creating your `.env` and includes a quick check to confirm it's actually being ignored before you commit anything.

## Setup

### 1. Postgres

```bash
brew install postgresql@16
brew services start postgresql@16
/opt/homebrew/opt/postgresql@16/bin/createdb analyst_agent
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Don't have an OpenAI API key yet? Get one at
[platform.openai.com/api-keys](https://platform.openai.com/api-keys):
log in (or create an account), click **Create new secret key**, then copy it
immediately — OpenAI only shows it to you once.

Copy the env template and fill in your real OpenAI API key:

```bash
cp .env.example .env
```

Then edit `backend/.env` and replace `sk-...` with your actual `OPENAI_API_KEY`.
`DATABASE_URL` is already set to the local Postgres database created above.

**If you're new to git, check that it's actually being ignored** before you
commit anything, by running this from the repo root:

```bash
git status
```

`backend/.env` should **not** appear anywhere in that output — not staged,
not untracked, nothing. If it's simply missing from the list, that's correct;
git is already ignoring it. If it ever *does* show up, stop — don't run
`git add` on it or commit — and double check the file is actually named
`.env` (not `.env.txt` or similar).

### 3. Frontend

Requires Node 20+. If you use `nvm`, switch to the version pinned in
`.nvmrc` before installing:

```bash
cd frontend
nvm use
npm install
```

(If `nvm use` reports the version isn't installed, run `nvm install` first.)

## Running the app

Start the backend (from `backend/`, with its venv):

```bash
./.venv/bin/uvicorn app.main:app --reload --port 8000
```

Start the frontend (from `frontend/`, in a separate terminal):

```bash
nvm use
npm run dev
```

Open **http://localhost:5173** in your browser. The frontend expects the
backend at `http://localhost:8000`.

## Notes

- Always invoke the backend's Python via `./.venv/bin/python` (or
  `./.venv/bin/uvicorn`, `./.venv/bin/pip`) rather than bare `python` — some
  shells have a `python` alias that bypasses the virtualenv entirely.
- If you write to `chroma_data/` (the vector store) from a separate script
  while the backend server is already running, restart the server afterward —
  it caches ChromaDB state in memory and won't see external writes otherwise.
