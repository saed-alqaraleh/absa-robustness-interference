from __future__ import annotations
from .hashing import hash_object

SIGNATURE_FIELDS = (
    "dataset_hash","split_hash","transformation_hash","source_selection_hash",
    "model_name","model_revision","tokenizer_name","tokenizer_revision",
    "seed","hyperparameters","code_version"
)

def compute_run_signature(meta: dict) -> str:
    missing = [k for k in SIGNATURE_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"missing run signature fields: {missing}")
    return hash_object({k:meta[k] for k in SIGNATURE_FIELDS})
