from __future__ import annotations

import csv
import hashlib
import shlex
from pathlib import Path
from typing import Iterable

CORE_CONDITIONS = [
    "Clean-Natural", "Clean-Matched", "Clean+RevTgt", "Clean+RevNon",
    "Clean+AddDiff", "Clean+Mixed",
]
LOPO_CONDITIONS = [
    "Clean+RevTgt+RevNon",
    "Clean+RevTgt+AddDiff",
    "Clean+RevNon+AddDiff",
]
CORE_EVALS = ("Clean", "RevTgt", "RevNon", "AddDiff")


def safe_slug(s: str) -> str:
    return s.replace("/", "__").replace("+", "plus").replace("-", "_").replace(" ", "_")


def run_id(domain: str, model: str, condition: str, seed: int) -> str:
    short = hashlib.sha256(model.encode()).hexdigest()[:8]
    return f"{domain}__{safe_slug(condition)}__m{short}__seed{seed}"


def get_stage_design(cfg: dict, stage: str) -> dict:
    """Return the frozen campaign dimensions for a named stage.

    The protocol-mandated first LOPO continuation stage remains deliberately
    minimal: RoBERTa-base, three LOPO conditions, and the same three seeds used
    in the minimal decisive matrix (2 domains x 3 conditions x 3 seeds = 18).
    """
    if stage == "minimal":
        return {
            "models": [cfg["models"]["minimal_decisive"]],
            "seeds": list(cfg["training"]["seeds_minimal"]),
            "conditions": list(CORE_CONDITIONS),
        }
    if stage == "full":
        return {
            "models": list(cfg["models"]["final"]),
            "seeds": list(cfg["training"]["seeds_final"]),
            "conditions": list(CORE_CONDITIONS),
        }
    if stage == "lopo":
        return {
            "models": [cfg["models"]["minimal_decisive"]],
            "seeds": list(cfg["training"]["seeds_minimal"]),
            "conditions": list(LOPO_CONDITIONS),
        }
    raise ValueError(f"Unknown campaign stage: {stage}")


def build_plan(
    domains: Iterable[str],
    models: Iterable[str],
    conditions: Iterable[str],
    seeds: Iterable[int],
    model_pins: dict | None = None,
) -> list[dict]:
    rows = []
    model_pins = model_pins or {}
    for d in domains:
        for m in models:
            for c in conditions:
                for seed in seeds:
                    rows.append({
                        "run_id": run_id(d, m, c, int(seed)),
                        "dataset": d,
                        "model": m,
                        "model_revision": (model_pins.get(m) or {}).get("revision"),
                        "training_condition": c,
                        "seed": int(seed),
                        "status": "PLANNED",
                    })
    return rows


def write_plan_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty campaign plan")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_train_command(row: dict, root: str | Path, hp: dict, stage: str = "minimal") -> str:
    root = Path(root)
    d = row["dataset"]
    c = row["training_condition"]
    rid = row["run_id"]

    if stage == "lopo" and c not in LOPO_CONDITIONS:
        raise ValueError(f"Non-LOPO condition {c!r} supplied to LOPO campaign")

    train = root / "transformations" / "materialized" / d / c / "train.jsonl"
    dev = root / "data_processed" / d / "dev.jsonl"
    data_root = root / "data_processed" / d
    eval_paths = {
        "Clean": data_root / "test.jsonl",
        "RevTgt": data_root / "evaluation" / "RevTgt.jsonl",
        "RevNon": data_root / "evaluation" / "RevNon.jsonl",
        "AddDiff": data_root / "evaluation" / "AddDiff.jsonl",
    }
    out = root / "runs" / stage / rid
    transformation_manifest = (
        root / "transformations" / "materialized" / d / "LOPO_MANIFEST.json"
        if stage == "lopo"
        else root / "manifests" / d / "transformation_manifest.json"
    )

    args = [
        "/usr/bin/python3", "scripts/05_train_run.py",
        "--train", str(train),
        "--dev", str(dev),
        "--model", row["model"],
        "--condition", c,
        "--dataset", d,
        "--seed", str(row["seed"]),
        "--out-dir", str(out),
        "--max-length", str(hp.get("max_length", 128)),
        "--lr", str(hp.get("learning_rate", 2e-5)),
        "--batch", str(hp.get("train_batch_size", 16)),
        "--epochs", str(hp.get("epochs", 8)),
        "--pair-order", str(hp.get("pair_order", "sentence_first")),
        "--repo-root", str(root / "code"),
        "--split-manifest", str(root / "manifests" / d / "split_manifest.json"),
        "--transformation-manifest", str(transformation_manifest),
        "--source-selection", str(root / "audits" / "eligibility" / d / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"),
        "--master-manifest", str(root / "manifests" / "master_run_manifest.jsonl"),
    ]
    if row.get("model_revision"):
        args.extend([
            "--model-revision", row["model_revision"],
            "--tokenizer-revision", row["model_revision"],
        ])
    for name in CORE_EVALS:
        args.extend(["--eval", f"{name}={eval_paths[name]}"])
    return " ".join(shlex.quote(a) for a in args)
