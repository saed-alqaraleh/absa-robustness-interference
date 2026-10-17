# Evidence index: manuscript and supplementary material

The repository follows the manuscript *One-Sided Data Augmentation Pits Sensitivity against Invariance in Aspect-Based Sentiment Analysis*. All references to supplementary tables below use the numbering in the submitted supplementary PDF. This index distinguishes directly recomputable saved-prediction metrics from inference procedures requiring the accompanying statistical code.

| Supplementary table | Underlying study | Public evidence | Recheck command |
| --- | --- | --- | --- |
| S1: reciprocal interference and inferential support | I | `studies/study1_encoder_transfer/{bert_roberta_deberta,modernbert}/runs/`, `results/study1/` | `python scripts/verify_release.py --study 1` verifies accuracy and Macro-F1; confidence intervals and adjusted p-values require the statistical protocol |
| S2: edit-induced error | I | Study I paired cohort prediction records and related scientific code | `python scripts/verify_release.py --study 1` checks input metrics; specialized paired statistics require study code |
| S3: reversal co-exposure | I | Study I `bert_roberta_deberta/runs/lopo/` prediction records | `python scripts/verify_release.py --study 1` |
| S4: audit-based exclusion sensitivity | II–III | `results/study3/`, `results/study2/`; Study II and III prediction records | `python scripts/verify_release.py --study 2` and `--study 3`; raw third-party audit sentences are not redistributed |
| S5–S12: complete encoder results | I | All 174 run JSONs and 696 study predictions | `python scripts/verify_release.py --study 1` |
| S13: effects relative to Clean-Natural | I | Study I run JSONs and source-aligned saved predictions | `python scripts/verify_release.py --study 1` |
| S14: Qwen3-8B LoRA transfer matrix | IV | `studies/study4_lora_adaptation/runs/`, `results/study4/LORA_RESULTS.json` | `python scripts/verify_release.py --study 4` verifies saved accuracy endpoints; bootstrapped intervals use study code |
| S15: LoRA prespecified contrasts and co-exposure | IV | Same as S14; `results/study4/contrasts.csv` | `python scripts/verify_release.py --study 4` for underlying endpoints |
| S16: complete LoRA profiles | IV | All 24 fine-tuning plus two zero-shot records | `python scripts/verify_release.py --study 4` |

The manuscript's core quantitative evidence for Studies I–IV is independently recomputed from frozen predictions by the release verifier. Statistical bootstrap confidence intervals, model training, external model weights, and third-party review text are outside that lightweight verification command's scope. The original code required to reproduce analysis and inference workflows is grouped under the four `code/` trees, together with its recorded protocol where distributable.
