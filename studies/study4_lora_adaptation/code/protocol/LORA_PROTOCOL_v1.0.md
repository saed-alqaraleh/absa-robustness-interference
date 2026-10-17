# Study IV protocol v1.0: LoRA fine-tuning of a current open LLM (fixed before any run)

Date: 16 October 2026. This file is hashed (SHA-256) into every run signature; any change after the first
run makes completed runs refuse to be reused.

## Question

Does one-sided fine-tuning of a current instruction-tuned LLM reproduce the reciprocal
sensitivity–invariance interference that Study I found in fine-tuned encoders, under the same
exposure-matched design? Study III showed that six one-sided *demonstrations* do not; this study
changes the adaptation setting to parameter updates while keeping the model class (instruction LLM).

## Fixed design

| Item | Setting |
| --- | --- |
| Model | `Qwen/Qwen3-8B` (Apache-2.0, dense 8B). The Hugging Face commit resolved by the first run is pinned in `MODEL_REVISION.json`; every later run must load the same commit |
| Adaptation | LoRA: rank 16, alpha 32, dropout 0.05, on q, k, v, o, gate, up and down projections; base weights frozen in bfloat16 |
| Prompt | Study III zero-shot prompt, unchanged: the same instruction and domain line, one query, the model's chat template with thinking disabled |
| Training objective | Cross-entropy over the three label scores, where a label's score is the log-probability of its first answer token summed (logsumexp) over the surface forms used in Study III (`positive`, `Positive`, ` positive`, ` Positive`, and likewise for the other labels) |
| Prediction | argmax of the same three label scores (identical to Study III scoring) |
| Training data | The frozen Study I files `transformations/materialized/<domain>/<condition>/train.jsonl`, unchanged: Clean-Matched, Clean+RevTgt, Clean+RevNon, Clean+Mixed. Every file has the clean base plus B added records (B = 1004 Restaurant, 419 Laptop) |
| Recipe (not tuned) | AdamW, learning rate 1e-4, weight decay 0, linear warm-up over the first 10% of steps then linear decay, effective batch 16 (micro-batch 8, accumulation 2), at most 3 epochs, gradient clipping 1.0, gradient checkpointing |
| Checkpoint rule | Clean-development accuracy (`data_processed/<domain>/dev.jsonl`) after every epoch; the best epoch's adapter is kept; early stopping with patience 1. ARTS never selects anything |
| Seeds | 13, 37, 73 (seed-matched to Study I) |
| Runs | 4 conditions × 2 domains × 3 seeds = 24 fine-tuning runs, plus 2 zero-shot reference jobs (no training) |
| Evaluation | Clean test and the protected ARTS cohorts: Restaurant 1120 / 846 / 444 / 1120, Laptop 638 / 466 / 135 / 638 (Clean / RevTgt / RevNon / AddDiff) |

No hyperparameter is tuned: the recipe is a standard LoRA configuration fixed in advance, applied
identically to every condition, so differences between conditions cannot come from tuning.

## Endpoints (as Study I and II)

- Family accuracy R_j per run; T_ij = R_j(condition) − R_j(Clean-Matched), seed-matched.
- R_mean = mean of RevTgt, RevNon, AddDiff accuracy; W = their minimum (per run, then averaged).
- J_SI = share of the 444 / 135 ARTS sources with both a RevTgt and a RevNon item whose two items are
  both classified correctly.
- Seeds are averaged first, then domains with equal weight.

## Inference

- 10,000 resamples of ARTS sentence clusters within each domain (seed 20261007); the same resample
  is applied to every run, so all contrasts are paired. Trained seeds are held fixed; between-seed
  SD is reported separately. Percentile 95% intervals; two-sided bootstrap p-values.
- Holm correction over the four primary reciprocal contrasts.

## Hypotheses (fixed before any run; every result is reported)

- **L1 (primary, reciprocal interference).** T(RevTgt → RevNon) < 0 and T(RevNon → RevTgt) < 0 in
  each domain (four directed contrasts, Holm-adjusted). Seed-level signs (12 contrasts) are reported.
- **L2 (co-exposure).** Mixed minus Clean-Matched > 0 on R_mean, W and J_SI (equal-domain mean).
- **Descriptive.** Fine-tuned profiles against the zero-shot Qwen3-8B reference; in-family gains
  T_ii; change in R_mean for one-sided training.

**Reading of L1, fixed in advance.** "Reproduced" if all four directed means are negative and all
four Holm-adjusted intervals exclude zero; "partly reproduced" if all four means are negative but
not all intervals exclude zero; "not reproduced" otherwise. Because Study III used a different model
(Qwen2.5-7B) and six demonstrations, a contrast between this study and Study III separates adaptation
setting from model class only partly; the zero-shot Qwen3-8B reference is reported for that reason.

## Data protection

ARTS files and the Study I training files are read only. Outputs go to `<root>/lora_qwen3_v1/`.
