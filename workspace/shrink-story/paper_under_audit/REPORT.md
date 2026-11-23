# Integrity Forensics Report — paper-under-audit-

**Verdict:** ⚪ REVIEW_UNAVAILABLE (incomplete sweep — not an acquittal)  ·  **Observability:** L2  ·  **Taxonomy:** v0.5  ·  **Adjudicator:** deterministic-rules-v2

> This is decision SUPPORT for a human reviewer. It flags discrepancies to investigate — it does **not** judge misconduct. `CLEAN_GIVEN_EVIDENCE` means "nothing checkable at L2 is broken", not "the paper is honest".

> **How to read the table.** Every proposal an auditor made is listed, at the severity that auditor proposed. The verdict above applies a fixed rule to those proposals; it is not a ruling that they stand. The columns are what you weigh: `Anchored: NO` means the quoted span was not found in the paper (that proposal is already dropped to info); `Observability L2 ✗` means the auditor said it needs evidence this run did not have; `FP-risk: high` is the auditor's own estimate that it may be a false positive; `Surface` marks a presentation signal rather than an integrity claim; `Ext-check` means the auditor marked it unsettled itself.

## Findings (evidence first)

| ID | Dimension | Severity | Pattern | Where | Anchored | Observability | FP-risk | Surface | Ext-check | Contest |
|----|-----------|----------|---------|-------|----------|---------------|---------|---------|-----------|---------|
| — | — | no findings proposed | — | — | — | — | — | — | — | — |

### Detail


## Coverage

| Skill | Status |
|---|---|
| `adversarial-case-builder` | ⛔ review_unavailable |
| `ai-style-impressions` | ⛔ review_unavailable |
| `baseline-comparison-audit` | ⛔ review_unavailable |
| `citation-forensics` | ⛔ review_unavailable |
| `consistency-audit` | ⛔ review_unavailable |
| `eval-design-forensics` | ⛔ review_unavailable |
| `experiment-forensics` | ⛔ review_unavailable |
| `novelty-duplication-advisory` | ⛔ review_unavailable |
| `presentation-signals` | ⛔ review_unavailable |
| `proof-derivation-forensics` | ⛔ review_unavailable |

## Counts

- critical: 0  ·  major: 0  ·  minor: 0  ·  info: 0
- needing evidence this run lacked: 0  ·  dropped as unanchored: 0
- AI writing-style impressions (zero verdict weight): 0

## Limitations

- Deterministic-only run (no cross-model reviewer): semantic + code-level dimensions were NOT run.
- COVERAGE INCOMPLETE: reviewer unavailable for verdict-bearing dimension(s) baseline-comparison-audit, citation-forensics, consistency-audit, eval-design-forensics, experiment-forensics, presentation-signals, proof-derivation-forensics — this run is NOT a full sweep. A clean result here means 'nothing found in the dimensions that ran', never 'clean overall'.
- Zero-weight track 'adversarial-case-builder' reviewer unavailable — its report section is missing; the integrity verdict is unaffected (zero verdict weight).
- Zero-weight track 'ai-style-impressions' reviewer unavailable — its report section is missing; the integrity verdict is unaffected (zero verdict weight).
- Zero-weight track 'novelty-duplication-advisory' reviewer unavailable — its report section is missing; the integrity verdict is unaffected (zero verdict weight).

_Human review required: always. This report does not issue a verdict on misconduct._
