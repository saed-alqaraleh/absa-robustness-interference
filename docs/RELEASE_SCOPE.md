# Scope of evidence

The release includes source-text-free experimental predictions, study-level computation modules, recorded configs and experiments, machine-readable aggregate outputs, and a supplementary tables PDF. It does not contain raw third-party review sentences, in-context demonstrations, human-audit spreadsheets containing review text, adapters, base-model weights, raw training corpora, or internal editorial correspondence.

The public Study I predictions retain gold and predicted labels, source IDs and cluster IDs; textual aspect terms and extraneous columns were excluded from these public derivatives. Machine-specific paths were removed or normalized in derived run JSON and code files. `reproducibility/PUBLIC_DERIVATIVES.json` links the SHA-256 hashes of changed originals to changed public files. The checksums of released files are recorded separately.

Complete frozen-output arithmetic is independently verified by `scripts/verify_release.py`; published bootstrap analyses should be evaluated using the documented scientific implementation and the respective original protocol. No training run was repeated for the release.
