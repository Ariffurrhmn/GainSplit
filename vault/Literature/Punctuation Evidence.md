---
tags: [literature, preprocessing]
updated: 2026-11-09
---

# Punctuation Evidence

Should punctuation be removed before training? Evidence gathered 2026-11-09 via Consensus (abstracts only unless stated). Hub: [[00 - Project Context]]. Index: [[Literature Index]]. Decision: [[Decisions]].

## What Headlines sarcasm papers do
- Seed paper removes punctuation (VERIFIED from the PDF, [[Seed Paper - Irfan 2025]]). Whether before or after making helper labels is not stated.
- Other Headlines + BERT papers (Scola and Segura-Bedmar 2021; Dehkharghani 2025; Nandini 2026; Liu and Xie 2021; Sharma 2023; Karkiner and Sert 2024): abstracts do not say how punctuation is handled (UNVERIFIED; full texts not read). Winkle 2025 lists lowercasing and stop-word removal for a non-BERT model, punctuation not mentioned.
- So there is no visible community standard for this dataset (INFERRED from abstracts).

## Does removal matter for BERT-type models?
- Siino et al. 2023 (Inf. Syst., survey + experiments): preprocessing can change Transformer accuracy; best choice depends on task and dataset.
- Kurniasih and Manik 2022: preprocessing had insignificant effect on BERT-embedding models.
- Rezapour 2024 (preprint): removing punctuation and stop words hurt emotion classification with transformers.
- Miyajiwala et al. 2022 (preprint): BERT is sensitive to removing tokens such as punctuation.
- Alzahrani and Jololian 2021 (preprint): BERT best with no preprocessing (author profiling).
- Summary (INFERRED): effect is usually small and task-dependent; no general rule.

## Is punctuation a sarcasm cue?
- Yes in social media text: exclamations and intensifiers mark sarcasm (Liebrecht 2013; Kunneman 2015; Liebeskind and Baczkowska 2025); punctuation features help feature-based models (Son 2019; Gupta 2020).
- For Headlines, punctuation is also an **outlet style** marker (The Onion vs HuffPost editorial style), so it can act as a shortcut rather than sarcasm (IDEA; links to [[Risks]] item 4 and the source-masked test in [[Experiment Plan]]).

## Consequence for GainSplit (INFERRED)
Removal matches the seed, reduces one outlet-style shortcut, and the literature does not show a large cost for BERT. Keep-punctuation can be run as a small robustness check.
