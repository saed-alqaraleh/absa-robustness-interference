#!/usr/bin/env python3
"""Stage 1 preflight: does the Stage-1 trainer reproduce the Study-I Mixed results?

Compares the six regression-phase runs (CE on the Study-I Clean+Mixed file, seeds 13/37/73)
with the archived Study-I RoBERTa Mixed accuracies (identical to the Stage 0 reruns).
The Stage-1 trainer is a plain PyTorch loop, not the Hugging Face Trainer, so exact equality
is not expected: the two 3-seed means are independent draws. With s = max(original SD, 1.0 pp),
the difference of two 3-seed means has standard error about sqrt(2/3)*s, so a cell is within
tolerance when |mean difference| <= 2*sqrt(2/3)*s. The check passes when at least 7 of the 8
cells (domain x Clean/RevTgt/RevNon/AddDiff) are within tolerance (false-alarm rate about 5%
for a correct trainer). This is an implementation check run before any Stage-1 outcome exists;
it is not part of the protocol's pass rule.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.common import CONFIRMATORY_SEEDS, DOMAINS, EVAL_SETS, KIT, STAGE1_SUBDIR, atomic_write_json, log, read_csv, run_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    ref = {(r["domain"], r["evaluation_condition"], int(r["seed"])): 100 * float(r["value"])
           for r in read_csv(KIT / "reference" / "roberta_original_metrics.csv")
           if r["metric"] == "accuracy" and r["training_condition"] == "Clean+Mixed"}
    cells, missing = [], []
    for d in DOMAINS:
        for fam in EVAL_SETS:
            new = []
            for s in CONFIRMATORY_SEEDS:
                rd = root / STAGE1_SUBDIR / "runs" / "regression" / run_id("Mixed", d, s, {})
                if not (rd / "DONE.json").exists():
                    missing.append(rd.name)
                    continue
                new.append(100 * json.loads((rd / "run_result.json").read_text())["eval_summary"][fam]["accuracy"])
            if len(new) < len(CONFIRMATORY_SEEDS):
                continue
            old = [ref[(d, fam, s)] for s in CONFIRMATORY_SEEDS]
            tol = 2 * math.sqrt(2 / 3) * max(statistics.stdev(old), 1.0)
            delta = statistics.fmean(new) - statistics.fmean(old)
            cells.append({"domain": d, "family": fam, "original_mean": statistics.fmean(old), "stage1_mean": statistics.fmean(new),
                          "delta_pp": delta, "tolerance_pp": tol, "ok": abs(delta) <= tol})
            log("regress", f"{d:10s} {fam:8s} original {statistics.fmean(old):6.2f}  stage1 {statistics.fmean(new):6.2f}  "
                           f"delta {delta:+.2f} (tol {tol:.2f}) {'ok' if abs(delta) <= tol else 'OUT'}")
    if missing:
        log("regress", f"INCOMPLETE: {sorted(set(missing))}")
        return 2
    ok = sum(c["ok"] for c in cells) >= len(cells) - 1
    atomic_write_json(root / STAGE1_SUBDIR / "preflight" / "regression_check.json", {"pass": ok, "cells": cells})
    log("regress", "PASS: the Stage-1 trainer reproduces Study-I Mixed within tolerance" if ok else
        "FAIL: investigate the trainer before the tuning phase")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
