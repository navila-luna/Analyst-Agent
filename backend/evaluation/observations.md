# Eval observations log

Running notes from analyzing `run_eval.py` output against `test_cases.json`.
Goal: use real data (not guesses) to pick `DISTANCE_THRESHOLD` in `app/qa.py`.

---

## 2026-08-14 — first run-through

Raw distances from the first eval run:

**Answerable (from docs):**
| Question | dist |
|---|---|
| How do I request a new AWS permission? | 0.588 |
| Who owns the billing service? | 0.559 |
| What should I do to get access to the team's private Slack channel? | 0.586 |
| What is one of the goals for the first week of onboarding? | 0.593 |
| How is a deployment triggered? | 0.415 |
| What should I do if a deploy causes errors? | 0.516 |

**Unanswerable (out of scope):**
| Question | dist |
|---|---|
| What's the capital of France? | 0.983 |
| What's our company's parental leave policy? | 0.827 |
| How do I set up a VPN connection? | 0.913 |
| What's the weather forecast for tomorrow? | 0.873 |
| Who is the CEO of our biggest competitor? | 0.891 |

**Observation:**
-

**What this suggests about the threshold:**
-

---
