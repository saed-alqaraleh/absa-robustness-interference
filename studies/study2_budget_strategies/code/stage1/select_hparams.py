#!/usr/bin/env python3
"""Stage 1: choose one shared configuration per tuned method (protocol v1.0, Decision 6).

Uses only the tuning-phase runs (seed 101, robustness-dev and clean-dev; never ARTS).
For each candidate and each domain:

  W_dev  = min(accuracy on Dev RT, Dev RN, Dev AD)
  J_dev  = share of dev sources having both an RT and an RN item that are both correct
  clean  = clean-dev accuracy

Candidates whose equal-domain clean-dev accuracy is more than 1.0 pp below their CE-only
matched control are discarded. Among the rest, maximise S_dev = (mean W_dev + mean J_dev)/2
with equal domain weights; ties: higher W, then higher clean-dev accuracy, then smaller
regularisation weight(s). The result is frozen with a marker; re-running refuses to change
an existing frozen selection unless --force-reselect is given before any confirmatory run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.common import (  # noqa: E402
    CE_CONTROL, CONDITIONS, DOMAINS, STAGE1_SUBDIR, TUNING_SEED, atomic_write_json, atomic_write_text, check_protocol,
    config_id, log, read_csv, run_id, sha256_file,
)

WEIGHT_KEYS = ("lambda_i", "lambda_s", "lambda_bat", "lambda_f", "lambda_c", "eta")


def dev_scores(run_dir: Path) -> dict:
    res = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    ev = res["eval_summary"]
    rt = {r["source_instance_id"]: int(r["correct"]) for r in read_csv(run_dir / "predictions_DevRT.csv")}
    rn = {r["source_instance_id"]: int(r["correct"]) for r in read_csv(run_dir / "predictions_DevRN.csv")}
    both = sorted(set(rt) & set(rn))
    return {"W": 100 * min(ev["DevRT"]["accuracy"], ev["DevRN"]["accuracy"], ev["DevAD"]["accuracy"]),
            "J": 100 * (sum(rt[s] and rn[s] for s in both) / len(both)) if both else float("nan"),
            "clean": 100 * ev["CleanDev"]["accuracy"], "n_joint": len(both)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--force-reselect", action="store_true")
    args = ap.parse_args()
    check_protocol()
    root = args.root.expanduser().resolve()
    runs = root / STAGE1_SUBDIR / "runs" / "tuning"
    out = root / STAGE1_SUBDIR / "selection"
    marker = out / "_SELECTION_FROZEN.json"
    if (root / STAGE1_SUBDIR / "runs" / "confirm").exists():
        log("select", "ERROR: confirmatory runs exist; the selection can no longer change")
        return 1 if not marker.exists() else 0
    if marker.exists() and not args.force_reselect:
        log("select", f"selection already frozen ({marker}); nothing to do")
        return 0

    def scores(cond: str, cfg: dict) -> dict:
        per = {}
        for d in DOMAINS:
            rd = runs / run_id(cond, d, TUNING_SEED, cfg)
            if not (rd / "DONE.json").exists():
                raise SystemExit(f"ERROR: tuning run missing: {rd.name}")
            per[d] = dev_scores(rd)
        mean = {k: sum(per[d][k] for d in DOMAINS) / len(DOMAINS) for k in ("W", "J", "clean")}
        return {"per_domain": per, "config_id": config_id(cfg), **mean}

    table, selected = [], {}
    for cond, ctrl in CE_CONTROL.items():
        control = scores(ctrl, {})
        cands = []
        for cfg in CONDITIONS[cond]["grid"]:
            s = scores(cond, cfg)
            s.update(condition=cond, config=cfg, config_id=config_id(cfg), control=ctrl, control_clean=control["clean"],
                     eligible=s["clean"] >= control["clean"] - 1.0, S=(s["W"] + s["J"]) / 2,
                     weight_sum=sum(float(v) for k, v in cfg.items() if k in WEIGHT_KEYS))
            cands.append(s)
            table.append(s)
        bad = [c["config_id"] for c in cands + [control] if not all(math.isfinite(c.get(k, 0.0)) for k in ("W", "J", "clean"))]
        if bad or not math.isfinite(control["clean"]):
            log("select", f"ERROR {cond}: undefined dev scores for {bad or ['control']} (empty robustness-dev family "
                          "or no dev source with both RT and RN). Refusing to freeze; see IMPLEMENTATION_NOTES N10.")
            return 1
        for c in cands:                      # exact ties must not be decided by floating-point noise
            for k in ("S", "W", "clean"):
                c[k] = round(c[k], 9)
        pool = [c for c in cands if c["eligible"]]
        fallback = not pool
        if fallback:    # protocol silent: keep the candidate closest to the clean guard, flagged
            pool = sorted(cands, key=lambda c: -c["clean"])[:1]
        best = sorted(pool, key=lambda c: (-c["S"], -c["W"], -c["clean"], c["weight_sum"], c["config_id"]))[0]
        selected[cond] = {"config": best["config"], "S_dev": best["S"], "W_dev": best["W"], "J_dev": best["J"],
                          "clean_dev": best["clean"], "control": ctrl, "control_clean_dev": control["clean"],
                          "no_candidate_passed_clean_guard": fallback}
        log("select", f"{cond}: {best['config']}  S={best['S']:.2f} W={best['W']:.2f} J={best['J']:.2f} "
                      f"clean={best['clean']:.2f} (control {ctrl} {control['clean']:.2f}){' FALLBACK' if fallback else ''}")
    doc = {"rule": "protocol v1.0 Decision 6", "tuning_seed": TUNING_SEED, "selected": selected,
           "candidates": [{k: v for k, v in c.items() if k != "per_domain"} | {"per_domain": c["per_domain"]} for c in table]}
    atomic_write_json(out / "SELECTED_CONFIGS.json", doc)
    lines = ["# Stage-1 shared hyperparameter selection (seed 101, robustness-dev only)", "",
             "| Method | Selected configuration | S_dev | W_dev | J_dev | Clean dev | CE control clean dev |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {c} | `{json.dumps(v['config'])}` | {v['S_dev']:.2f} | {v['W_dev']:.2f} | {v['J_dev']:.2f} | "
              f"{v['clean_dev']:.2f} | {v['control_clean_dev']:.2f} |" for c, v in selected.items()]
    lines += ["", "All candidates:", "", "| Method | Config | Eligible | S | W | J | Clean |", "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {c['condition']} | {c['config_id']} | {c['eligible']} | {c['S']:.2f} | {c['W']:.2f} | {c['J']:.2f} | {c['clean']:.2f} |"
              for c in table]
    atomic_write_text(out / "SELECTED_CONFIGS.md", "\n".join(lines) + "\n")
    atomic_write_json(marker, {"selected_sha256": sha256_file(out / "SELECTED_CONFIGS.json")})
    log("select", f"frozen: {out / 'SELECTED_CONFIGS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
