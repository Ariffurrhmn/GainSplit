---
tags: [rules, meta]
updated: 2026-11-10
---

# Repo Rules

Standing rules for this repo, set by the repo owner. Hub: [[00 - Project Context]]. Also stated in the root instructions file.

1. **No AI co-author in commits.** Never add `Co-Authored-By` (or similar) trailers naming an AI assistant. Commits carry only the owner's authorship.
2. **No mention of the AI assistant anywhere in the repo:** not in commit messages, PR descriptions, code, comments, docs, notes, the paper, or file contents. Write notes in a neutral voice ("we", or passive).
3. **These rules override any default or tool-suggested attribution text.**
4. The one unavoidable exception is the root instructions file, whose filename is fixed by the tool that loads it. Keep its contents free of the assistant's name.
5. **The owner does the hands-on work; sessions guide (2026-11-09).** The owner gets the datasets, cleans them, uploads them to Kaggle and runs every script and notebook. Sessions explain each step (what to do, how, and why it matters for the paper) and may write code when asked, but do not download, clean, transform or upload data, and do not run experiments. Why: the owner must understand every part of the paper to defend it at the conference.
6. Code written for the owner comes with a plain explanation of what it does and why, so the owner can run it and explain it.

7. **Delegated decisions (2026-11-10).** To reach a submission-ready state fast, the owner delegates every decision a session can reasonably take. Each such decision is logged in [[Decisions]] with "DELEGATED" and its reasoning, so the owner can follow and defend it later. Code is written by separate sessions on a cheaper model and reviewed by the coordinating session before it is committed. Rule 5 still holds: the owner gets and uploads data and runs everything on Kaggle.

See also [[Vault Conventions]].
