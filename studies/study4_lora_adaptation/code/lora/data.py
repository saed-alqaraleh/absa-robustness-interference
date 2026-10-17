"""Read the frozen Study I training files, the clean dev split and the protected evaluation cohorts.

Everything here is read-only. Counts are checked against the frozen Study I sizes so that a wrong
folder or a modified file stops the run instead of silently changing the design.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from lora.common import (ARTS_FAMILIES, EXPECTED_EVAL, EXPECTED_TRAIN_ROWS, LABELS, dev_path, eval_paths, read_jsonl,
                         train_path)


def _check_row(r: dict, where: str) -> None:
    for k in ("sentence", "aspect_term", "label"):
        if k not in r:
            raise SystemExit(f"{where}: record without '{k}': keys {sorted(r)}")
    if r["label"] not in LABELS:
        raise SystemExit(f"{where}: unexpected label {r['label']!r}")


def load_train(root: Path, d: str, condition: str, expected_rows: int | None = None) -> list[dict]:
    p = train_path(root, d, condition)
    if not p.is_file():
        raise SystemExit(f"missing training file {p}")
    rows = read_jsonl(p)
    for r in rows:
        _check_row(r, str(p))
    want = EXPECTED_TRAIN_ROWS[d] if expected_rows is None else expected_rows
    if want and len(rows) != want:
        raise SystemExit(f"{p}: {len(rows)} rows, expected {want} (frozen Study I size)")
    return [{"sentence": r["sentence"], "aspect_term": r["aspect_term"], "label": r["label"]} for r in rows]


def load_dev(root: Path, d: str) -> list[dict]:
    p = dev_path(root, d)
    if not p.is_file():
        raise SystemExit(f"missing dev file {p}")
    rows = read_jsonl(p)
    for r in rows:
        _check_row(r, str(p))
    return [{"set": "Dev", "sentence": r["sentence"], "aspect_term": r["aspect_term"], "label": r["label"],
             "source_instance_id": r.get("source_instance_id", ""), "cluster_id": r.get("cluster_id", "")} for r in rows]


def load_eval(root: Path, d: str, check_counts: bool = True) -> list[dict]:
    """Clean test split + the three ARTS families, in file order (identical for every run)."""
    items = []
    counts = Counter()
    for name, p in eval_paths(root, d).items():
        if not p.is_file():
            raise SystemExit(f"missing evaluation file {p}")
        for r in read_jsonl(p):
            _check_row(r, str(p))
            if "source_instance_id" not in r:
                raise SystemExit(f"{p}: evaluation record without source_instance_id")
            items.append({"set": name, "sentence": r["sentence"], "aspect_term": r["aspect_term"], "label": r["label"],
                          "source_instance_id": r["source_instance_id"], "cluster_id": r.get("cluster_id", "")})
            counts[name] += 1
    if check_counts and dict(counts) != EXPECTED_EVAL[d]:
        raise SystemExit(f"{d}: evaluation counts {dict(counts)} differ from the protected cohorts {EXPECTED_EVAL[d]}")
    for f in ARTS_FAMILIES:
        ids = [it["source_instance_id"] for it in items if it["set"] == f]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{d}/{f}: source_instance_id is not unique; pairing would be ambiguous")
    return items


def label_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(r["label"] for r in rows))
