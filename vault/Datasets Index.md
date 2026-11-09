---
tags: [index, datasets]
updated: 2026-11-08
---

# Datasets Index

Hub: [[00 - Project Context]]. Used in [[Experiment Plan]].

| Note | Role | Licence |
|---|---|---|
| [[Headlines Dataset]] | Main, 1-100% budget curve | Kaggle licence not verified |
| [[iSarcasmEval]] | Transfer / evaluation | Repo MIT |

Others considered: SARC (LREC 2018; ~1.3M self-labelled Reddit; "/s" labels noisy; terms NOT verified); SemEval-2018 Task 3 (CC BY-NC-SA 4.0, shared on request for academic use; VERIFIED repo).

Local status (2026-11-08): both datasets above are downloaded under `data/` (see each note). Not downloaded and not needed as data: SARC, SemEval-2018 (optional extras). Teacher models (cardiffnlp sentiment, GoEmotions) are models from the Hugging Face hub, fetched at run time; lexicons (VADER via pip) likewise.
