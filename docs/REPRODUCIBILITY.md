# Reproducibility

## Reconstruction from frozen predictions (no GPU)

From the repository root on Python 3.10 or newer:

```bash
python scripts/verify_release.py
python scripts/checksums.py --verify
```

The first command recomputes saved accuracy, three-class Macro-F1 (Studies I-II), evaluation accuracies (III-IV), and mean, worst-family and paired joint-satisfaction endpoints (IV). It checks against the archived run JSON and job-completion records, and checks Study III source CSV signatures. Nonzero exit indicates a failed check. The frozen summaries and published inferential procedures can be inspected in `results/` and the scientific scripts under `studies/`. Exact recreation of all bootstrap confidence intervals is not claimed by the lightweight verifier.

## Repeating training/inference

Each study has its original implementation modules and (where available) author-documented environment files in its `code/` folder. Model weights, adapters, third-party training/evaluation text, and some original model-execution inputs are external. Run the applicable training scripts only after obtaining those inputs and validating the hardware/software stack. The package verification has not rerun the original model training or inference. Absolute Drive-specific defaults in a small number of Study I scripts were replaced with a generic `./data` path in the public code derivatives.

## Recorded computational inventory

Study I: 174 runs, 696 cohort predictions. Study II: 6 regression-check runs, 62 tuning runs, 54 confirmatory runs; 488 cohort predictions. Study III: 64 completed inference jobs. Study IV: 24 trained LoRA runs and two zero-shot jobs; adapter weights external.

The exposed input records are complete for the above frozen-output checks. Excluded third-party review text is not needed to calculate the saved metrics.
