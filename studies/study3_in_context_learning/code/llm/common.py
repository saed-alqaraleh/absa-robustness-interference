"""LLM probe of reciprocal robustness interference on ARTS (kit v1.0.0).

Question: do instruction-tuned LLMs show the same sensitivity/invariance asymmetry as
fine-tuned encoders, and does *in-context* exposure to one edit type harm the other, as
fine-tuning on it does (Study I)? No training is involved.

Conditions (k = 6 demonstrations, matched across conditions):
  ZS       zero-shot
  Clean    6 clean training examples                    (exposure-matched control)
  RT       6 target-reversal examples (label flipped)   (sensitivity contract only)
  RN       6 non-target-reversal examples (label kept)  (invariance contract only)
  Mixed    2 RT + 2 RN + 2 AD from different sources    (co-exposure, unpaired)
  Triplet  2 sources x (RT, RN, AD), adjacent            (co-exposure, same-source)
Each demonstration condition is repeated with 3 demonstration draws (seeds 13, 37, 73).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

DOMAINS = ("restaurant", "laptop")
EVAL_SETS = ("Clean", "RevTgt", "RevNon", "AddDiff")
CONDITIONS = ("ZS", "Clean", "RT", "RN", "Mixed", "Triplet")
DEMO_SEEDS = (13, 37, 73)
K = 6
LABELS = ("negative", "neutral", "positive")
OUT_SUBDIR = "llm_probe_v1"
SCORING_VERSION = "label-first-token-logsumexp/v1"
DEFAULT_MODELS = ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3")

INSTRUCTION = ("Classify the sentiment expressed toward the given aspect term in the sentence. "
               "Only the opinion about that aspect matters; ignore opinions about other aspects. "
               "Answer with exactly one word: positive, negative, or neutral.")


def domain_word(d: str) -> str:
    return {"restaurant": "restaurant reviews", "laptop": "laptop reviews"}[d]


def item_text(r: dict) -> str:
    return f"Sentence: {r['sentence']}\nAspect: {r['aspect_term']}\nSentiment:"


def user_message(d: str, demos: list[dict], query: dict | None) -> str:
    s = f"{INSTRUCTION}\nThe sentences are {domain_word(d)}.\n\n"
    if demos:
        s += "Examples:\n\n" + "".join(item_text(x) + f" {x['label']}\n\n" for x in demos) + "Now classify this one.\n\n"
    if query is not None:
        s += item_text(query)
    return s


def jobs(models, domains=DOMAINS) -> list[dict]:
    out = []
    for m in models:
        for d in domains:
            out.append({"model": m, "domain": d, "condition": "ZS", "seed": 0})
            for c in CONDITIONS[1:]:
                for s in DEMO_SEEDS:
                    out.append({"model": m, "domain": d, "condition": c, "seed": s})
    return out


def select_jobs(all_jobs: list[dict], shard: str = "0/1", subshard: str = "0/1") -> list[dict]:
    """Job k belongs to notebook shard k % N; within it, the r-th job belongs to worker r % M."""
    i, n = (int(x) for x in shard.split("/"))
    j, m = (int(x) for x in subshard.split("/"))
    if not (0 <= i < n and 0 <= j < m):
        raise SystemExit(f"bad shard {shard} / subshard {subshard}")
    mine = [x for k, x in enumerate(all_jobs) if k % n == i]
    return [x for r, x in enumerate(mine) if r % m == j]


def slug(model: str) -> str:
    return model.strip("/").replace("/", "__")


def job_id(j: dict) -> str:
    return f"{j['domain']}__{j['condition']}__seed{j['seed']}"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def read_jsonl(p) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


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
