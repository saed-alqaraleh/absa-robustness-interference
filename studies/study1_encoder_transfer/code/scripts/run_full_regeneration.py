#!/usr/bin/env python
"""Regenerate every processed training input of Study I from the public raw data.

Chain (per domain):
  01_prepare_from_xml       -> <out>/data_processed/<d>/pre_firewall/{train,dev,test}.jsonl
  01b_content_firewall      -> <out>/data_processed/<d>/{train,dev,test}.jsonl  (+ exclusions, audit)
  01c_validate_towe         -> <out>/data_processed/towe_validated/<d>/train_validated.tsv
  02_align_towe             -> <out>/transformations/opinion_alignment/<d>/train/opinion_records.jsonl
  04_materialize_conditions -> <out>/transformations/materialized/<d>/<condition>/train.jsonl

The source selection (``FINAL_GENERATION_VALID_SOURCE_SELECTION.json`` per domain) is a
released, hash-verified input. WordNet 3.0 must be installed (``nltk.download('wordnet')``);
the generator stops with an error if it is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
XML = {"restaurant": ("Restaurants_Train_v2.xml", "Restaurants_Test_Gold.xml", "14res"),
       "laptop": ("Laptop_Train_v2.xml", "Laptops_Test_Gold.xml", "14lap")}
CONDS = ("Clean-Natural", "Clean-Matched", "Clean+RevTgt", "Clean+RevNon", "Clean+AddDiff", "Clean+Mixed")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str], log: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(CODE / "src"), PYTHONUNBUFFERED="1")
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(cmd[1:3]), "...", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=str(CODE), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        raise SystemExit(f"step failed ({rc}): {tail}; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semeval-dir", required=True)
    ap.add_argument("--towe-data-dir", required=True)
    ap.add_argument("--selection-dir", required=True, help="contains <domain>/FINAL_GENERATION_VALID_SOURCE_SELECTION.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260913)
    a = ap.parse_args()
    from nltk.corpus import wordnet as wn  # fail early and loudly
    if wn.get_version() != "3.0":
        raise SystemExit(f"WordNet 3.0 required, found {wn.get_version()}")
    out, py = Path(a.out_dir).resolve(), sys.executable
    sem, towe, sel = Path(a.semeval_dir).resolve(), Path(a.towe_data_dir).resolve(), Path(a.selection_dir).resolve()
    hashes: dict[str, str] = {}
    for d, (tr, te, t) in XML.items():
        dp = out / "data_processed" / d
        run([py, "scripts/01_prepare_from_xml.py", "--domain", d, "--train-xml", str(sem / tr),
             "--test-xml", str(sem / te), "--out-dir", str(dp / "pre_firewall")], out / "logs" / f"01_{d}.log")
        run([py, "scripts/01b_content_firewall.py", "--in-dir", str(dp / "pre_firewall"), "--out-dir", str(dp),
             "--domain", d, "--audit-out", str(out / "audits" / f"{d.upper()}_CONTENT_FIREWALL_AUDIT.json")],
            out / "logs" / f"01b_{d}.log")
    run([py, "scripts/01c_validate_towe.py",
         *sum((["--domain", f"{d}={towe / XML[d][2] / 'train.tsv'}="
                f"{out / 'data_processed' / 'towe_validated' / d / 'train_validated.tsv'}"] for d in XML), []),
         "--manifest-out", str(out / "manifests" / "towe_validated_release_manifest.json")], out / "logs" / "01c.log")
    for d in XML:
        dp = out / "data_processed" / d
        ops = out / "transformations" / "opinion_alignment" / d / "train"
        run([py, "scripts/02_align_towe.py", "--examples", str(dp / "train.jsonl"),
             "--towe-tsv", str(out / "data_processed" / "towe_validated" / d / "train_validated.tsv"),
             "--out-dir", str(ops)], out / "logs" / f"02_{d}.log")
        run([py, "scripts/04_materialize_conditions.py", "--train", str(dp / "train.jsonl"),
             "--opinions", str(ops / "opinion_records.jsonl"),
             "--selected-ids", str(sel / d / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"),
             "--out-dir", str(out / "transformations" / "materialized" / d),
             "--manifest-out", str(out / "manifests" / d / "transformation_manifest.json"), "--seed", str(a.seed)],
            out / "logs" / f"04_{d}.log")
        for name in ("train", "dev", "test"):
            hashes[f"data_processed/{d}/{name}.jsonl"] = sha(dp / f"{name}.jsonl")
        hashes[f"data_processed/towe_validated/{d}/train_validated.tsv"] = \
            sha(out / "data_processed" / "towe_validated" / d / "train_validated.tsv")
        for c in CONDS:
            hashes[f"transformations/materialized/{d}/{c}/train.jsonl"] = \
                sha(out / "transformations" / "materialized" / d / c / "train.jsonl")
    (out / "REGENERATION_HASHES.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("done:", out / "REGENERATION_HASHES.json")


if __name__ == "__main__":
    main()
