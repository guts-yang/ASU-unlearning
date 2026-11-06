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
Candidates: 200. Ground-truth donors: 2. Synthetic: 198. Excluded: 0.
Exclusion reasons: none.
Ground-truth rows keep a same-split donor answer. Synthetic rows replace the located slot with an ordinary personal name, a real city, or a calendar date, and leave y_minus empty.
Token comparison is descriptive whitespace, not Llama-2 alignment. No model weights were loaded.
Code version: 7ff37cd.
Command: `python3 -m workspace.I2.commands.build_same_split_qminus`.

Synthetic slots now use ordinary personal names, real city names, and calendar dates. Examples: Mateo Cruz, Leila Haddad, Naledi Mokoena, Fez, Helsinki, the 6th of September, 1972. `y_minus` stays empty. The two ground-truth rows are unchanged.
