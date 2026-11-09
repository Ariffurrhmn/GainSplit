# GainSplit

Where does the gain come from? Separating adversarial training, multi-task helper heads, and helper-label quality in semi-supervised sarcasm detection.

Target: IEEE QPAIN 2027 (status: proposal accepted; building code and setup).

## Layout

| Path | Contents |
|------|----------|
| `vault/` | Obsidian vault: project memory, proposal, literature, decisions (open as a vault in Obsidian) |
| `docs/proposal/` | Proposal A PDF |
| `src/` | Model and training code |
| `experiments/` | Experiment configs and run scripts |
| `data/` | Datasets (git-ignored) |
| `results/` | Metrics, tables, figures |
| `paper/` | Manuscript source |

## Data

Datasets are not stored in git. Place them under `data/`:

| Dataset | Source | Put at |
|---|---|---|
| News Headlines Sarcasm (Misra and Arora) | https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection (`Sarcasm_Headlines_Dataset.json`) | `data/headlines/` |
| iSarcasmEval (Abu Farha et al., 2022) | https://github.com/iabufarha/iSarcasmEVAL | `data/isarcasmeval/` |

Check each dataset's licence and cite the original papers before reuse.
