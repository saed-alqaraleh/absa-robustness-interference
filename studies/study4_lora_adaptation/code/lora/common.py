"""Shared constants, paths and helpers for Study IV (LoRA fine-tuning of Qwen3-8B).

The prompt and the label-scoring rule are copied unchanged from the Study III kit (llm_kit_v1,
llm/common.py and llm/run_llm.py) so that zero-shot, in-context and fine-tuned results are scored
identically. The training files are the frozen Study I files, so exposure matches Study I exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

KIT_VERSION = "1.0.0"
DOMAINS = ("restaurant", "laptop")
EVAL_SETS = ("Clean", "RevTgt", "RevNon", "AddDiff")
ARTS_FAMILIES = ("RevTgt", "RevNon", "AddDiff")
LABELS = ("negative", "neutral", "positive")
VARIANTS = {"negative": ("negative", "Negative", " negative", " Negative"),
            "neutral": ("neutral", "Neutral", " neutral", " Neutral"),
            "positive": ("positive", "Positive", " positive", " Positive")}

CONDITIONS = ("Clean-Matched", "Clean+RevTgt", "Clean+RevNon", "Clean+Mixed")
ZS = "ZS"                      # zero-shot reference: no training
SEEDS = (13, 37, 73)
MODEL_ID = "Qwen/Qwen3-8B"
OUT_SUBDIR = "lora_qwen3_v1"

RECIPE = {"learning_rate": 1e-4, "weight_decay": 0.0, "warmup_ratio": 0.1, "schedule": "linear",
          "micro_batch": 8, "grad_accum": 2, "max_epochs": 3, "patience": 1, "max_grad_norm": 1.0,
          "max_prompt_tokens": 256, "base_dtype": "bfloat16", "gradient_checkpointing": True,
          "objective": "cross-entropy over the three label scores", "selection": "best clean-dev accuracy"}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}

# Frozen Study I sizes (manuscript Table 3 and Section 3.3); any mismatch stops the run.
EXPECTED_TRAIN_ROWS = {"restaurant": 4455, "laptop": 2581}
EXPECTED_EVAL = {"restaurant": {"Clean": 1120, "RevTgt": 846, "RevNon": 444, "AddDiff": 1120},
                 "laptop": {"Clean": 638, "RevTgt": 466, "RevNon": 135, "AddDiff": 638}}
EXPECTED_J_SOURCES = {"restaurant": 444, "laptop": 135}

INSTRUCTION = ("Classify the sentiment expressed toward the given aspect term in the sentence. "
               "Only the opinion about that aspect matters; ignore opinions about other aspects. "
               "Answer with exactly one word: positive, negative, or neutral.")

KIT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = KIT_DIR / "protocol" / "LORA_PROTOCOL_v1.0.md"


def domain_word(d: str) -> str:
    return {"restaurant": "restaurant reviews", "laptop": "laptop reviews"}[d]


def item_text(r: dict) -> str:
    return f"Sentence: {r['sentence']}\nAspect: {r['aspect_term']}\nSentiment:"


def user_message(d: str, query: dict) -> str:
    """Study III zero-shot user message (llm_kit_v1 user_message with no demonstrations)."""
    return f"{INSTRUCTION}\nThe sentences are {domain_word(d)}.\n\n" + item_text(query)


# ----------------------------------------------------------------------------- paths
def train_path(root: Path, d: str, condition: str) -> Path:
    return root / "transformations" / "materialized" / d / condition / "train.jsonl"


def dev_path(root: Path, d: str) -> Path:
    return root / "data_processed" / d / "dev.jsonl"


def eval_paths(root: Path, d: str) -> dict[str, Path]:
    return {"Clean": root / "data_processed" / d / "test.jsonl",
            **{f: root / "data_processed" / d / "evaluation" / f"{f}.jsonl" for f in ARTS_FAMILIES}}


def out_root(root: Path) -> Path:
    return root / OUT_SUBDIR


def run_name(j: dict) -> str:
    return f"{j['domain']}__{j['condition']}__seed{j['seed']}"


def jobs() -> list[dict]:
    """Fixed job order: zero-shot references first, then seed-major so each shard gets a mix."""
    out = [{"domain": d, "condition": ZS, "seed": 0} for d in DOMAINS]
    for s in SEEDS:
        for d in DOMAINS:
            for c in CONDITIONS:
                out.append({"domain": d, "condition": c, "seed": s})
    return out


def select_jobs(all_jobs: list[dict], shard: str) -> list[dict]:
    i, n = (int(x) for x in shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad --shard {shard}")
    return [j for k, j in enumerate(all_jobs) if k % n == i]


# ----------------------------------------------------------------------------- io
def read_jsonl(p) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def log(stage: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{stage}] {msg}", flush=True)


def code_digests() -> dict[str, str]:
    """Digests of the code that defines a run; part of every run signature."""
    files = ["lora/common.py", "lora/data.py", "lora/scoring.py", "lora/train_lora.py"]
    return {f: sha256_file(KIT_DIR / f) for f in files}


def protocol_sha256() -> str:
    return sha256_file(PROTOCOL)
