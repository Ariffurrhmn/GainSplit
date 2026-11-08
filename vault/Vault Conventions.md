---
tags: [meta]
---

# Vault Conventions

Purpose: let a new session get up to speed in minutes. Hub: [[00 - Project Context]]. Rules for the repo: [[Repo Rules]].

1. **One idea per note.** Link with `[[Note Name]]` (shortest-path names, no folders).
2. **Every note** has frontmatter: `tags`, `created`, `updated`. Update `updated` on edit.
3. **Every note links to a hub** (usually [[00 - Project Context]]) and to at least one related note. No orphans.
4. **Evidence labels:** VERIFIED / INFERRED / IDEA / UNVERIFIED. Put the source URL next to VERIFIED claims.
5. **Where things go:**
   - Paper read: `Literature/<Short name>.md` (template: [[Templates/Literature Note]]), then a row in [[Literature Index]].
   - Dataset: `Datasets/<Name>.md`, row in [[Datasets Index]].
   - A choice made: append to [[Decisions]] (date, choice, why).
   - Unanswered thing: [[Open Questions]]; once answered, move the answer into the relevant note.
   - Parked idea: [[00 - Backlog]]; promising: [[Research Ideas]].
   - Experiment results: raw files in `results/`, summary note linked from [[Experiment Plan]].
6. **End of every session:** add an entry at the top of [[Session Log]] ([[Templates/Session Entry]]) and refresh "Current state" in the hub.
7. Never delete a note others link to; mark it `status: superseded` and link the replacement.
8. Follow [[Repo Rules]] (no assistant attribution or mentions).
