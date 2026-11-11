---
tags: [results, pass4]
created: 2026-11-11
---

# Pairgen Pilot Results

Hub: [[00 - Project Context]]. Notebook `experiments/04_pairgen_pilot.ipynb`, commit 11ee9ae, run by the owner on Kaggle 2026-11-11. Files in `results/pairgen_pilot/`. All numbers below read from those files (VERIFIED); readings are INFERRED.

## Numbers
- 200 train seeds (100 per class), 67/67/66 to Qwen2.5-7B / Gemma-2-9B / Mistral-7B-v0.3.
- Speed: 0.61 / 0.80 / 1.24 s per seed (batch 16), load 113-152 s. Full split (7,600 seeds per model) about 1.3 / 1.7 / 2.6 h: each model fits one Kaggle session.
- Automatic checks pass for 100% (Gemma, Mistral) and 97% (Qwen: 2 copies of the seed). No refusals. Gemma in float16 on T4 works.
- Word overlap between the two sides (Jaccard) about 0.22: topic kept, wording mostly rewritten.
- Outlet-cue rate (top-100 sarcastic cue words): real sarcastic 0.37 vs real serious 0.05; generated sarcastic 0.18-0.23 vs generated serious 0.21-0.26. **The outlet-word gap is gone in generated pairs.** Both generated sides lean to formal HuffPost-like wording (serious-cue rate about 0.4).
- Judge (Phi-3.5-mini) on 200 real seeds: accuracy 0.635; serious 0.80, sarcastic 0.47. It answers SERIOUS for most inputs. On generated: serious sides 0.91-0.97 agree, sarcastic sides 0.33-0.37.
- Kept pairs (checks + judge on both sides): 26-33%.

## Reading
- Generation works and is fast; the judge does not. Phi misses half the real sarcastic headlines, so the low keep rate mostly reflects the judge, not the generators. Phi is not usable as a filter.
- By eye, sarcastic sides are mixed: some clearly satirical, many only mildly ironic or plain. Serious sides are good.
- Next: owner's blind check of `audit_sheet.csv` (100 items) gives the ground truth; replace the judge (candidate: cross-family voting with the generators themselves, calibrated on real headlines) and strengthen the sarcasm instruction.
