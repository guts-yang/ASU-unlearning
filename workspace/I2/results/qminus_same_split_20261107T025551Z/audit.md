# Batch A task 1 pair sidecar

This directory is an auditable candidate list and exclusion list. It is not approved for localization.

Source: `/Users/chenyliao/Repositories/Machine_Unlearning/ASU-unlearning/Right-to-be-forgotten/data/tofu/forget05.json`
SHA-256 before and after: `a8c207a905af73ea4797b1af92a14cf458f42df567c941f7284f87d930096f4b`
Bytes before and after: 603987
Source rows: 2000. Task counts: {'1': 200, '2': 200, '3': 200, '4': 200, '5': 200, '6': 200, '7': 200, '8': 200, '9': 200, '10': 200}.
Scope: task 1. Calibration entities: 5. Holdout entities: 5.
Entity ids are local 20-row blocks after possessive-normalized names, not dataset ids.
Names from this scope found in other tasks: none.
Author interval is unstable because each side is below 8.
Scope stays a descriptive pilot. Confirmatory localization needs a new data version and preregistration, enough independent authors and pair clusters, and a cross-task entity and donor-split check.
Record status counts: generated=165, excluded=33, pending_review=2, approved_for_localization=0.
Primary single-slot candidates: 2. Compound place+date diagnostics: 2. Synthetic non-primary: 163.
Primary pair clusters: 1. Compound diagnostic clusters: 1.
Exclusion reasons: same_relation_ambiguous=30, same_relation_missing=3.
A primary q- keeps the q+ sentence and replaces one author name. y- is the donor's original answer for the same relation, even when the donor question is worded differently. No donor-sentence alignment is required. Rows without one verifiable same-relation answer are excluded and are not given a guessed answer.
The place+date rows that name the author from a birth place and a birth date stay compound diagnostics. They are not in the first single-slot set.
Synthetic rows keep an empty y_minus and are not counted as candidates.
Token comparison is descriptive whitespace, not Llama-2 alignment. No model weights were loaded.
Code version: e23dda7.
Command: `python3 -m workspace.I2.commands.build_same_split_qminus`.
