# One-Sided Data Augmentation Pits Sensitivity against Invariance in Aspect-Based Sentiment Analysis

Research implementations, frozen prediction evidence, and supplementary analyses accompanying the manuscript *One-Sided Data Augmentation Pits Sensitivity against Invariance in Aspect-Based Sentiment Analysis*.

## Scientific scope

This study examines how one-sided aspect-sentiment augmentation changes competing behavioral requirements: target-reversal sensitivity and non-target invariance. It reports controlled encoder-level transfer, budget-matched training strategies, in-context demonstrations, and LoRA adaptation, with protected ARTS evaluation cohorts.

| Study | Contents | Recorded evaluation |
| --- | --- | --- |
| I: Encoder transfer | BERT, RoBERTa, DeBERTa-v3, ModernBERT frozen records | 174 runs, 696 cohort files |
| II: Training strategies | Regression checks, tuning and confirmatory results | 122 runs, 488 cohort files |
| III: In-context learning | Qwen2.5-7B-Instruct / Mistral-7B-Instruct results | 64 inference jobs |
| IV: LoRA adaptation | Qwen3-8B evaluation and per-run outputs | 24 fine-tunes + 2 zero-shot jobs |

## Quick start

Clone the repository and run the offline verification suite (Python 3.10+, standard library only):

```bash
git clone https://github.com/saed-alqaraleh/absa-robustness-interference.git
cd absa-robustness-interference
python scripts/verify_release.py
python scripts/checksums.py --verify
```

## Verify the released frozen evidence

The first program checks accuracy and Macro-F1 of available Study I-II cohort predictions, 64 Study III inference outputs and their recorded hashes, and accuracy, mean-family, worst-family and joint-satisfaction endpoints of 26 Study IV outputs. The second checks SHA-256 file integrity. No training, model downloads, API calls, or GPU are required.

See [reproducibility instructions](docs/REPRODUCIBILITY.md) for scope and dependencies; [data access](docs/DATA_SOURCES.md) for benchmark licensing; [release scope](docs/RELEASE_SCOPE.md) for omitted text and checkpoint materials. The [evidence index](docs/EVIDENCE_INDEX.md) maps each supplementary table to its archived records and verification scope. The full [Supplementary Tables S1-S16](tables/Supplementary_Tables_S1-S16.pdf) accompany the numerical results.

The saved-result checks do not establish that independently retraining pretrained models will yield identical weights. The release intentionally does not redistribute third-party benchmark text or large model adapters.

## Repository contents

| Directory | Purpose |
| --- | --- |
| [`studies/study1_encoder_transfer/`](studies/study1_encoder_transfer/) | Encoder transfer and controlled augmentation; recorded scripts and predictions |
| [`studies/study2_budget_strategies/`](studies/study2_budget_strategies/) | Budget strategy comparisons, protocol, runs and analyses |
| [`studies/study3_in_context_learning/`](studies/study3_in_context_learning/) | In-context learning inference records and analysis code |
| [`studies/study4_lora_adaptation/`](studies/study4_lora_adaptation/) | LoRA adaptation source modules and frozen evaluation records |
| [`results/`](results/) | Machine-readable study summaries |
| [`tables/`](tables/) | Supplementary Tables S1–S16 (PDF) |
| [`scripts/`](scripts/) | Offline numerical and checksum verification |
| [`docs/`](docs/) | Data access, evidentiary mapping and reproducibility scope |
| [`reproducibility/`](reproducibility/) | Release hashes and original-to-public derivative trace |

## Release and citation

The initial verified research package is published as [version 1.0.0](https://github.com/saed-alqaraleh/absa-robustness-interference/releases/tag/v1.0.0) (17 October 2026). The release includes the four-study experimental archive, machine-readable results, supplementary Tables S1–S16, and verification scripts.

**Cite the research software and supplementary evidence as:**

Alqaraleh, S. (2026). *One-Sided Data Augmentation Pits Sensitivity against Invariance in Aspect-Based Sentiment Analysis: Research Software and Supplementary Evidence* (Version 1.0.0) [Research software and supplementary materials]. GitHub. https://github.com/saed-alqaraleh/absa-robustness-interference/releases/tag/v1.0.0

Machine-readable and reference-manager formats are provided in [CITATION.cff](CITATION.cff) and [CITATION.bib](CITATION.bib). The accompanying journal article should be cited separately using its published bibliographic record.

