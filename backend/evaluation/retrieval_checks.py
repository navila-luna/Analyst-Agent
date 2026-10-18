"""Run a small, human-readable retrieval evaluation without calling OpenAI."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.search import search_documents


SIMILARITY_THRESHOLD = 0.25


@dataclass(frozen=True)
class RetrievalCase:
    category: str
    question: str
    expected_chunks: tuple[str, ...] = ()
    answerable: bool | None = True


CASES = [
    # Direct questions
    RetrievalCase("direct", "How do I roll back a deployment?", ("runbook_deploy.md::0",)),
    RetrievalCase("direct", "Where do I request repository access?", ("onboarding.md::0",)),
    RetrievalCase("direct", "How do I request a new AWS permission?", ("faq.md::0",)),
    RetrievalCase("direct", "Where are ADRs stored?", ("faq.md::1",)),
    RetrievalCase("direct", "What should I do before my first deployment?", ("onboarding.md::1",)),

    # Paraphrases
    RetrievalCase("paraphrase", "How can I undo a bad release?", ("runbook_deploy.md::0",)),
    RetrievalCase("paraphrase", "Who can give me access to the backend code?", ("onboarding.md::0",)),
    RetrievalCase("paraphrase", "Where does the team document architecture choices?", ("faq.md::1",)),
    RetrievalCase("paraphrase", "The latest release broke production. How can I restore the earlier version?", ("runbook_deploy.md::0",)),
    RetrievalCase("paraphrase", "How do I get permission to view the service repositories?", ("onboarding.md::0",)),

    # Questions requiring more than one piece of evidence
    RetrievalCase("multi-step", "How do I deploy to production, and what should I do if it fails?", ("runbook_deploy.md::0", "runbook_deploy.md::1")),
    RetrievalCase("multi-step", "Explain the production promotion, rollback, and escalation process.", ("runbook_deploy.md::0", "runbook_deploy.md::1")),
    RetrievalCase("multi-step", "As a new engineer, how do I get repository access and prepare for my first deployment?", ("onboarding.md::0", "onboarding.md::1")),

    # Conditional questions
    RetrievalCase("conditional", "How do I roll back a deployment?", ("runbook_deploy.md::0",)),
    RetrievalCase("conditional", "The rollback did not fix the incident. What do I do next?", ("runbook_deploy.md::1",)),
    RetrievalCase("conditional", "The rollback has been running for two minutes. Should I escalate?", ("runbook_deploy.md::0", "runbook_deploy.md::1")),
    RetrievalCase("conditional", "The rollback failed ten minutes ago. Who should I contact?", ("runbook_deploy.md::1",)),

    # Ambiguous questions require human review rather than an automatic pass.
    RetrievalCase("ambiguous", "How do I get access?", answerable=None),
    RetrievalCase("ambiguous", "What should I do when it fails?", answerable=None),
    RetrievalCase("ambiguous", "Where is the documentation?", answerable=None),
    RetrievalCase("ambiguous", "Who owns it?", answerable=None),
    RetrievalCase("ambiguous", "How do I deploy?", answerable=None),

    # Related vocabulary, but no complete answer in the current documents
    RetrievalCase("related-unsupported", "Can I deploy without approval?", answerable=False),
    RetrievalCase("related-unsupported", "How do I become the primary on-call engineer?", answerable=False),
    RetrievalCase("related-unsupported", "How do I create a new ADR?", answerable=False),
    RetrievalCase("related-unsupported", "What happens if PagerDuty is unavailable?", answerable=False),
    RetrievalCase("related-unsupported", "Can contractors receive GitHub access?", answerable=False),
    RetrievalCase("related-unsupported", "How do I increase the home-office stipend?", answerable=False),
    RetrievalCase("related-unsupported", "How do I change the weekly on-call schedule?", answerable=False),

    # Broad questions triggered by terminology found in the documents
    RetrievalCase("terminology-trap", "How do I use Docker?", answerable=False),
    RetrievalCase("terminology-trap", "Where is the platform?", answerable=False),
    RetrievalCase("terminology-trap", "How do I manage Python packages?", answerable=False),
    RetrievalCase("terminology-trap", "How do I create an incident?", answerable=False),
    RetrievalCase("terminology-trap", "What does main mean?", answerable=False),
    RetrievalCase("terminology-trap", "How do I configure Slack?", answerable=False),

    # False premises should retrieve evidence that can correct them.
    RetrievalCase("contradiction", "Production deployment is automatic, right?", ("runbook_deploy.md::0",)),
    RetrievalCase("contradiction", "Rollbacks take ten minutes to propagate, correct?", ("runbook_deploy.md::0",)),
    RetrievalCase("contradiction", "ADRs are stored in the internal wiki, right?", ("faq.md::1",)),
    RetrievalCase("contradiction", "Repository access is requested through PagerDuty, correct?", ("onboarding.md::0",)),

    # User role and circumstances
    RetrievalCase("role-dependent", "I am a new engineer. What should I do before my first deploy?", ("onboarding.md::1",)),
    RetrievalCase("role-dependent", "I am the primary on-call engineer and rollback failed. What comes next?", ("runbook_deploy.md::1",)),
    RetrievalCase("role-dependent", "I am a contractor. How do I get repository access?", answerable=False),
    RetrievalCase("role-dependent", "My manager has not approved my production promotion. Can I proceed?", ("runbook_deploy.md::0",)),

    # Similar questions around the current threshold boundary
    RetrievalCase("threshold-boundary", "How do I request access to the backend repositories?", ("onboarding.md::0",)),
    RetrievalCase("threshold-boundary", "How do I get access to the backend repositories?", ("onboarding.md::0",)),
    RetrievalCase("threshold-boundary", "Where can I find the backend code?", ("onboarding.md::0",)),
    RetrievalCase("threshold-boundary", "I need permission to view the platform services.", ("onboarding.md::0",)),
    RetrievalCase("threshold-boundary", "Who grants access to the service source code?", ("onboarding.md::0",)),
]


def chunk_id(result: dict) -> str:
    return f"{result['doc_name']}::{result['chunk_index']}"


def assess(case: RetrievalCase, results: list[dict], threshold: float) -> tuple[str, str]:
    top_score = results[0]["similarity"] if results else None

    if case.answerable is None:
        return "REVIEW", "ambiguous question; inspect whether clarification is needed"

    if case.answerable is False:
        if top_score is None or top_score < threshold:
            return "PASS", f"best score is below {threshold:.2f}"
        return "FLAG", f"unsupported question scored at or above {threshold:.2f}"

    ranks = {chunk_id(result): rank for rank, result in enumerate(results, start=1)}
    scores = {chunk_id(result): result["similarity"] for result in results}
    missing = [expected for expected in case.expected_chunks if expected not in ranks]
    if missing:
        return "MISS", "missing " + ", ".join(missing)

    low_scoring = [
        expected for expected in case.expected_chunks
        if scores[expected] < threshold
    ]
    if low_scoring:
        details = ", ".join(
            f"{expected}={scores[expected]:.2f}" for expected in low_scoring
        )
        return "FLAG", f"expected evidence is below {threshold:.2f}: {details}"

    if len(case.expected_chunks) == 1 and ranks[case.expected_chunks[0]] != 1:
        expected = case.expected_chunks[0]
        return "FLAG", f"{expected} was found, but only at rank {ranks[expected]}"

    found = ", ".join(
        f"{expected} at rank {ranks[expected]}" for expected in case.expected_chunks
    )
    return "PASS", found


def run(category: str | None, threshold: float, details: bool) -> int:
    selected = [case for case in CASES if category is None or case.category == category]
    if not selected:
        categories = ", ".join(sorted({case.category for case in CASES}))
        print(f"Unknown category: {category}\nAvailable categories: {categories}")
        return 2

    counts = {"PASS": 0, "FLAG": 0, "MISS": 0, "REVIEW": 0}

    for case in selected:
        results = search_documents(case.question)
        status, explanation = assess(case, results, threshold)
        counts[status] += 1
        top_score = results[0]["similarity"] if results else None
        score_text = "n/a" if top_score is None else f"{top_score:.2f}"

        print(f"[{status:6}] {case.category:19} top={score_text}  {case.question}")
        print(f"         {explanation}")

        if details:
            for rank, result in enumerate(results, start=1):
                print(
                    f"         {rank}. {chunk_id(result):28} "
                    f"similarity={result['similarity']:.2f}"
                )
        print()

    print("Summary")
    print("-------")
    print(f"Cases:  {len(selected)}")
    for status in ("PASS", "FLAG", "MISS", "REVIEW"):
        print(f"{status.title():7}: {counts[status]}")
    print(f"Threshold: {threshold:.2f}")
    print("PASS/FLAG/MISS evaluate retrieval only; REVIEW requires human judgment.")

    return 1 if counts["MISS"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        help="Run one category, such as direct, paraphrase, or related-unsupported.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold used for checks (default: {SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print all five ranked chunks for every question.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.category, args.threshold, args.details))
