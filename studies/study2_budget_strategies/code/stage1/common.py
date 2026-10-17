"""Shared constants, paths and helpers for the Stage-1 SCCT pilot (protocol v1.0, 7 Oct 2026)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

KIT = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = KIT / "protocol" / "Stage1_SCCT_Protocol_v1.0_2026-10-07.md"
PROTOCOL_SHA256 = "073bf0bf72e5de38ba93e2eaa98e68ebaa1bbaebe04d739f7fc7e73bf45b3621"

DOMAINS = ("restaurant", "laptop")
HISTORICAL_BUDGET = {"restaurant": 1004, "laptop": 419}
GENERATOR_SEED = 20260913          # frozen transformation seed (Study I)
CONSTRUCTION_KEY = "20261007"      # protocol date; keys every deterministic source ordering
TUNING_SEED = 101
CONFIRMATORY_SEEDS = (13, 37, 73)
LABELS = ("negative", "neutral", "positive")
LABEL2ID = {k: i for i, k in enumerate(LABELS)}
POLAR_FLIP = {"positive": "negative", "negative": "positive"}
DEGREE_ADVERBS = ("very", "really", "extremely")   # the frozen RevNon intensification operator
ARTS_FAMILIES = ("RevTgt", "RevNon", "AddDiff")
EVAL_SETS = ("Clean",) + ARTS_FAMILIES

STAGE1_SUBDIR = "stage1_v1"        # all Stage-1 outputs live under <root>/stage1_v1
MIN_TUNING_ITEMS = 30              # minimum RN items and RT&RN sources per domain in the tuning set
RESERVE_FRACTION = 0.15            # amendment v1.1 option: share of the pool held out for tuning
AMENDMENTS_FILE = KIT / "protocol" / "AMENDMENTS.json"


def amendments() -> dict:
    """Pre-outcome implementation switches. v1.0 defaults; a v1.1 amendment edits protocol/AMENDMENTS.json."""
    cfg = {"version": "1.0", "tuning_set": "dev", "fei_infonce": False}
    if AMENDMENTS_FILE.exists():
        cfg.update(json.loads(AMENDMENTS_FILE.read_text(encoding="utf-8")))
    if cfg["tuning_set"] not in ("dev", "dev+reserved"):
        raise SystemExit(f"bad tuning_set in {AMENDMENTS_FILE}")
    return cfg


FEI_INFONCE = bool(amendments()["fei_infonce"])   # False = protocol v1.0 form (note N20)

# Study-I training recipe (fixed across every Stage-1 method)
RECIPE = {"model": "FacebookAI/roberta-base", "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
          "learning_rate": 2e-5, "batch_size": 16, "eval_batch_size": 32, "epochs": 8, "max_length": 128,
          "warmup_ratio": 0.1, "weight_decay": 0.01, "max_grad_norm": 1.0, "fp16": True,
          "early_stopping_patience": 2, "pair_order": "sentence_first"}

# Conditions. "data" names the added-record construction; "grouped" means the clean anchor and the
# derivatives of a source share a batch; "grid" lists the protocol's tuning grid (empty = untuned).
CONDITIONS: dict[str, dict[str, Any]] = {
    "CleanMatched": {"data": "clean_matched", "method": "ce", "grouped": False, "grid": [{}], "decisive": "reference"},
    "Mixed": {"data": "mixed", "method": "ce", "grouped": False, "grid": [{}], "decisive": "core"},
    "BAT": {"data": "bat", "method": "bat", "grouped": True,
            "grid": [{"lambda_bat": v, "margin": 1.0} for v in (0.25, 0.5, 1.0)], "decisive": "core"},
    "Fei": {"data": "fei", "method": "fei", "grouped": True,
            "grid": [{"mu": mu, "lambda_f": lf} for mu in (0.05, 0.10, 0.20) for lf in (0.25, 0.5, 1.0)],
            "decisive": "core"},
    "Paired": {"data": "triplet", "method": "ce", "grouped": True, "grid": [{}], "decisive": "ablation"},
    "SCCT": {"data": "triplet", "method": "scct", "grouped": True,
             "grid": [{"lambda_i": li, "lambda_s": ls, "margin": 1.0} for li in (0.25, 0.5, 1.0) for ls in (0.25, 0.5, 1.0)],
             "decisive": "proposed"},
    "Consistency": {"data": "mixed", "method": "consistency", "grouped": True,
                    "grid": [{"lambda_c": v} for v in (0.25, 0.5, 1.0)], "decisive": "secondary"},
    "GroupDRO": {"data": "mixed", "method": "groupdro", "grouped": False,
                 "grid": [{"eta": v} for v in (0.01, 0.1, 1.0)], "decisive": "secondary"},
    "PCGrad": {"data": "mixed", "method": "pcgrad", "grouped": False, "grid": [{}], "decisive": "secondary"},
}
# CE-only matched control used by the clean-dev guard in the shared selection rule (Decision 6)
CE_CONTROL = {"SCCT": "Paired", "BAT": "BAT_CE", "Fei": "Fei_CE", "Consistency": "Mixed", "GroupDRO": "Mixed"}
TUNING_ONLY_CONTROLS = {"BAT_CE": {"data": "bat", "method": "ce", "grouped": True},
                        "Fei_CE": {"data": "fei", "method": "ce", "grouped": True}}


def stage1_budget(historical: int, pool_size: int) -> int:
    """Protocol 3.2: largest multiple of six <= min(B_D, 2|P_D|)."""
    return 6 * (min(historical, 2 * pool_size) // 6)


def ordered(ids: Iterable[str], key: str) -> list[str]:
    """Deterministic, outcome-independent ordering of source ids."""
    return sorted(ids, key=lambda s: hashlib.sha256(f"{CONSTRUCTION_KEY}|{key}|{s}".encode()).hexdigest())


def config_id(cfg: dict) -> str:
    return "default" if not cfg else "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))


def run_id(condition: str, domain: str, seed: int, cfg: dict) -> str:
    return f"{condition}__{domain}__{config_id(cfg)}__seed{seed}"


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else [])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def log(stage: str, msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{stage}] {msg}", flush=True)


def check_protocol() -> None:
    actual = sha256_file(PROTOCOL_FILE)
    if actual != PROTOCOL_SHA256:
        raise SystemExit(f"protocol file digest {actual} != frozen {PROTOCOL_SHA256}; refusing to run")
