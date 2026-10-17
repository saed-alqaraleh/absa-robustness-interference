#!/usr/bin/env python
"""Validate the TOWE training TSV before alignment.

A row is excluded when the token stream embedded in ``target_tags`` or
``opinion_words_tags`` differs from the row's tokenized sentence. Excluded rows are
listed in the manifest; nothing is corrected by hand. Kept lines (and the header) are
copied byte-for-byte, so a file without bad rows is an exact copy of the original.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stream(field: str) -> list[str]:
    return [t.rsplit("\\", 1)[0] for t in field.split()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", action="append", required=True, help="NAME=RAW_TSV=VALIDATED_TSV; repeat")
    ap.add_argument("--manifest-out", required=True)
    a = ap.parse_args()
    manifest = {"policy": "Rows with internal token-stream/sentence inconsistencies are excluded automatically "
                          "from the auxiliary TOWE alignment resource. No manual correction or adjudication is performed.",
                "domains": {}}
    for spec in a.domain:
        name, raw, val = spec.split("=", 2)
        raw_p, val_p = Path(raw), Path(val)
        with raw_p.open(encoding="utf-8", newline="") as fh:
            lines = fh.read().splitlines(keepends=True)
        header = lines[0].rstrip("\r\n").split("\t")
        col = {c: i for i, c in enumerate(header)}
        keep, excluded = [lines[0]], []
        for no, line in enumerate(lines[1:], start=2):
            f = line.rstrip("\r\n").split("\t")
            toks = f[col["sentence"]].split()
            reasons = []
            if stream(f[col["target_tags"]]) != toks:
                reasons.append("target_token_stream_differs_from_sentence")
            if stream(f[col["opinion_words_tags"]]) != toks:
                reasons.append("opinion_token_stream_differs_from_sentence")
            if reasons:
                excluded.append({"line_no": no, "s_id": f[col.get("s_id", 0)], "sentence": f[col["sentence"]],
                                 "reasons": reasons})
            else:
                keep.append(line)
        val_p.parent.mkdir(parents=True, exist_ok=True)
        with val_p.open("w", encoding="utf-8", newline="") as fh:
            fh.write("".join(keep))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        manifest["domains"][name] = {"original_sha256": sha(raw_p), "original_rows": len(lines) - 1,
                                     "validated_sha256": sha(val_p), "validated_rows": len(keep) - 1,
                                     "excluded_rows": excluded}
        print(name, "excluded lines:", [e["line_no"] for e in excluded])
    Path(a.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest_out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
