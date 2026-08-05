# Thinking Like an Engineer: Reading Existing Code Critically

[BACKEND_WALKTHROUGH.md](BACKEND_WALKTHROUGH.md) covers *what order* to read
files in. This doc covers a different skill: once you can follow what code
*does*, how do you decide if it's actually **good**?

**The core lens: scalability.** Does a design's cost stay flat, or grow, as
the data grows? Three questions to run on almost anything in this codebase:

| Ask this | Because |
|---|---|
| **Speed** — does it get slower as documents/users/requests grow? | That's the delay a user actually feels. |
| **Storage** — does memory/disk grow right along with the data? | This is where storage bills and slow I/O sneak up. |
| **Fixed vs. proportional cost** — does it redo the *whole* thing even when only a tiny piece changed? | Most scalability bugs hide right here. |

## The 4-step read

1. **Name the problem first.** What breaks, or gets too slow/expensive,
   without this code? Can't say it in one sentence? You're not ready to
   judge the code yet.
2. **Read the code with that problem in mind.** You're checking "does this
   solve what I just named" — not reading blind.
3. **Hunt for flaws through the scalability lens.** What happens at 10x the
   data? 100x? Most flaws in a young codebase aren't logic bugs — they're
   spots that scale worse than whoever wrote it first assumed.
4. **Ask why it's built this way — or ask "why not X?"** A limitation might
   be a documented tradeoff (see `DISTANCE_THRESHOLD` in
   [qa.py:14-19](backend/app/qa.py)) — a real decision worth understanding
   before touching. Or it might just be the first thing that worked, never
   revisited. If the code doesn't explain itself, say the question out loud
   instead of assuming there's a reason.

## Worked example: manifest.py's hashing granularity

**The problem:** re-indexing shouldn't redo expensive work on content that
hasn't changed. Embedding text (running `all-MiniLM-L6-v2`) is the slow,
costly step here — not hashing.

**The code:** [manifest.py](backend/app/manifest.py) +
[index.py](backend/app/index.py) use two tiers of change detection:

- A **whole-file hash** — cheap first filter.
- A **per-chunk hash** — for files that changed, narrows down to only the
  specific chunks that are actually different.

**The flaw, through the scalability lens:**

- **Storage** — hashing per-chunk instead of per-file means the manifest
  grows with *total chunks*, not file count (a single long doc can be
  hundreds of chunks). [manifest.py:40-51](backend/app/manifest.py) also
  reads and rewrites the *entire* manifest as one JSON blob every run — no
  partial read/write. That's a fixed tax that grows with the corpus, no
  matter how small the actual change was.
- **Speed** — the per-chunk design is supposed to limit re-embedding to just
  the changed chunks. But chunking
  ([ingest.py:26-34](backend/app/ingest.py)) slides a fixed-size window over
  raw character positions, not sentence/paragraph boundaries. Edit the
  *middle* of a file and change its length by even one character, and every
  chunk after that point shifts — so nearly the whole back half of the file
  gets flagged "changed" and re-embedded anyway. The optimization really
  only pays off for two cases: same-length edits, and appending to the end.
  A mid-document edit — arguably the more common one — barely beats the
  "just re-embed everything" approach it was meant to replace.

**Why it's built this way, and what to ask instead:** a coarser, file-level
design has a real flaw — re-embedding an entire large file for a one-line
edit — and that's enough to justify going finer-grained. But the fix above
shows it's only a *partial* one: it mainly wins on tail edits and appends.
The sharper question: **why not make chunk boundaries content-aware (split
on paragraphs/sentences) instead of fixed character counts?** That would
make the optimization work for mid-document edits too, since an edit would
only ever invalidate the chunks whose content actually changed.

## Don't guess — measure

This is exactly what the Phase 6 eval pipeline ([PLAN.md](PLAN.md)) should
answer with data: how often do real edits land in the "good" case (tail
edits/appends) vs. the "degraded" case (mid-document edits)? Rare
mid-document edits mean the current design earns its complexity. Common
ones are real evidence for content-aware chunking — not just an opinion.

Either way: write down what you measured and why, the way the
`DISTANCE_THRESHOLD` comment in [qa.py:14-19](backend/app/qa.py) does. That
note is what lets the next person start from "here's what we learned"
instead of re-deriving it from scratch.
