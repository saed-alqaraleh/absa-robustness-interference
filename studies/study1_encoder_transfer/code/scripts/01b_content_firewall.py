#!/usr/bin/env python
"""Content firewall (leakage guard) applied after 01_prepare_from_xml.

Removes any record whose normalized sentence also occurs in a higher-priority split.
Priority: protected test > frozen dev > train. Normalization: Unicode NFKC, casefold,
whitespace collapse. Kept records are written byte-for-byte as they appear in the input
(no re-serialization), so the output is reproducible. No manual decisions are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def read_lines(p: Path) -> list[tuple[str, dict]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return [(line, json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, help="01_prepare_from_xml output (train/dev/test.jsonl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--audit-out", required=True)
    a = ap.parse_args()
    src, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {s: read_lines(src / f"{s}.jsonl") for s in ("train", "dev", "test")}
    before = {s: len(v) for s, v in data.items()}
    protected_test = {norm(r["sentence"]) for _, r in data["test"]}
    kept, excluded = {"test": data["test"]}, []
    kept["dev"] = []
    for line, r in data["dev"]:
        if norm(r["sentence"]) in protected_test:
            excluded.append((r, ["duplicate_of_protected_test"]))
        else:
            kept["dev"].append((line, r))
    frozen_dev = {norm(r["sentence"]) for _, r in kept["dev"]}
    kept["train"] = []
    for line, r in data["train"]:
        n = norm(r["sentence"])
        reasons = (["duplicate_of_protected_test"] if n in protected_test else []) + \
                  (["duplicate_of_frozen_dev"] if n in frozen_dev else [])
        if reasons:
            excluded.append((r, reasons))
        else:
            kept["train"].append((line, r))
    for s in ("train", "dev", "test"):
        (out / f"{s}.jsonl").write_text("".join(line for line, _ in kept[s]), encoding="utf-8", newline="")
    rows = []
    for r, reasons in excluded:
        rows.append({**r, "firewall_reasons": reasons,
                     "normalized_sentence_sha256": hashlib.sha256(norm(r["sentence"]).encode("utf-8")).hexdigest()})
    (out / "firewall_exclusions.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    remaining = {
        "train_test": len({norm(r["sentence"]) for _, r in kept["train"]} & protected_test),
        "train_dev": len({norm(r["sentence"]) for _, r in kept["train"]} & frozen_dev),
        "dev_test": len(frozen_dev & protected_test),
    }
    audit = {"domain": a.domain, "normalization": "Unicode NFKC + casefold + whitespace collapse",
             "priority": "protected_test > frozen_dev > train", "before_counts": before,
             "after_counts": {s: len(v) for s, v in kept.items()},
             "excluded_records": rows, "remaining_cross_split_normalized_overlap": remaining}
    Path(a.audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.audit_out).write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"domain": a.domain, "before": before, "after": audit["after_counts"],
                      "excluded_ids": [r["source_instance_id"] for r in rows], "remaining_overlap": remaining}))
    if any(remaining.values()):
        raise SystemExit("firewall left cross-split overlap; refusing to continue")


if __name__ == "__main__":
    main()
