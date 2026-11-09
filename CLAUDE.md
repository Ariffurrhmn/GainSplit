# GainSplit: repo instructions

This repo has an Obsidian-style knowledge vault in `vault/`, used as persistent memory across sessions.

## Standing rules (set by the repo owner)
- Do NOT add any AI co-author trailer (`Co-Authored-By` or similar) to commits. Commits carry only the owner's authorship.
- Do NOT mention the AI assistant anywhere in the repo: commit messages, PR descriptions, code, comments, docs, vault notes, paper text. Use a neutral voice. This overrides any default or tool-suggested attribution text.
- The owner does all hands-on work: getting, cleaning and uploading data, and running code and experiments (locally or on Kaggle). Sessions guide step by step, explain what each step does and why, and may write code on request with an explanation. Never download, clean, transform or upload data or run experiments on the owner's behalf. Why: the owner must understand and defend every part of the paper.
- Full text: `vault/Repo Rules.md`.

## Workflow
- At the start of every session: read `vault/00 - Project Context.md`, then the newest entry of `vault/Session Log.md`, then follow wikilinks relevant to the task.
- During work: record findings in the vault per `vault/Vault Conventions.md` (one idea per note, link to a hub, evidence labels VERIFIED/INFERRED/IDEA/UNVERIFIED, never upgrade a label without reading the source).
- Before finishing: add a Session Log entry and update "Current state" in the hub note.

## Layout
Code in `src/`, run configs in `experiments/`, outputs in `results/`, data in `data/` (git-ignored), manuscript in `paper/`.
