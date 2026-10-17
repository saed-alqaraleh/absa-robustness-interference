# Data access and redistribution

This repository contains author-generated prediction records, model outputs, run-level configurations, and statistical summaries. It deliberately does not redistribute original third-party SemEval-2014 or ARTS review text, protected audit sentences, in-context demonstration sentences, or original model weights.

- SemEval-2014 Task 4 (restaurant/laptop): https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
- ARTS evaluation resources: https://github.com/zhijing-jin/ARTS_TestSet
- Qwen, Mistral, BERT-family model weights must be obtained separately from their upstream providers under their terms.

The `source_instance_id` and `cluster_id` fields preserve source alignment in the public numeric evidence. No source review sentences are included. Public Study I CSV files are non-text derivatives of original recorded predictions: review-target strings and original local file paths are not redistributed.

Training from scratch requires the original upstream licensed data, the study-specific preparation procedure, appropriate access to model revisions, and substantial compute. The saved-prediction verification workflow does not require any of these.
