#!/usr/bin/env python3
"""Stage 1: build a run plan for one phase, or run it (resumable, live progress).

Phases (in this order):
  regression  6 runs   CE on the Study-I Mixed file, seeds 13/37/73. Checks this trainer
                       against the Stage 0 reruns; no Stage-1 outcome is involved.
  tuning      62 runs  seed 101 only; every grid point of BAT, Fei, SCCT, Consistency and
                       GroupDRO on both domains, plus the CE-only matched controls needed by the
                       clean-dev guard. Evaluates the robustness-dev set only, never ARTS.
  confirm     54 runs  9 conditions x 2 domains x seeds 13/37/73 with the shared configuration
                       selected by select_hparams.py. Requires the selection marker.

Usage:
  python stage1/run_plan.py plan --root R --phase tuning
  python stage1/run_plan.py run  --root R --phase tuning [--keep-going]

A run whose DONE.json matches is skipped, so re-running after a disconnect resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.common import (  # noqa: E402
    CE_CONTROL, CONDITIONS, CONFIRMATORY_SEEDS, DOMAINS, KIT, STAGE1_SUBDIR, TUNING_SEED, atomic_write_json,
    check_protocol, log, run_id, sha256_file,
)

SAVE_MODEL_CONFIRM = {"CleanMatched", "Mixed", "Paired", "SCCT"}


def load_frozen_selection(root: Path) -> dict:
    """The selection may only be used if it still matches the hash recorded when it was frozen."""
    sel = root / STAGE1_SUBDIR / "selection" / "SELECTED_CONFIGS.json"
    marker = root / STAGE1_SUBDIR / "selection" / "_SELECTION_FROZEN.json"
    if not marker.exists() or not sel.exists():
        raise SystemExit("ERROR: no frozen selection; run select_hparams.py after the tuning phase")
    if sha256_file(sel) != json.loads(marker.read_text(encoding="utf-8"))["selected_sha256"]:
        raise SystemExit("ERROR: SELECTED_CONFIGS.json differs from the frozen hash; refusing to use it")
    return json.loads(sel.read_text(encoding="utf-8"))["selected"]


def plan(root: Path, phase: str, python: str, skip_preflight: bool = False) -> list[dict]:
    base = root / STAGE1_SUBDIR / "runs" / phase
    runs = []

    def add(condition: str, domain: str, seed: int, cfg: dict, extra: list[str] | None = None) -> None:
        rid = run_id(condition, domain, seed, cfg)
        out = base / rid
        cmd = [python, "-u", str(KIT / "stage1" / "trainer.py"), "--root", str(root), "--condition", condition,
               "--domain", domain, "--seed", str(seed), "--config", json.dumps(cfg, sort_keys=True),
               "--phase", phase, "--out-dir", str(out)] + (extra or [])
        runs.append({"run_id": rid, "condition": condition, "domain": domain, "seed": seed, "config": cfg,
                     "out_dir": str(out), "command": cmd})

    if phase == "tuning" and not skip_preflight:
        pf = root / STAGE1_SUBDIR / "preflight" / "regression_check.json"
        if not pf.exists() or not json.loads(pf.read_text(encoding="utf-8")).get("pass"):
            raise SystemExit("ERROR: the regression preflight has not passed (run the regression phase and "
                             "regression_check.py first, or pass --skip-preflight-check deliberately)")
    if phase == "regression":
        for d in DOMAINS:
            for s in CONFIRMATORY_SEEDS:
                add("Mixed", d, s, {})
    elif phase == "tuning":
        for d in DOMAINS:
            for cond in ("SCCT", "BAT", "Fei", "Consistency", "GroupDRO"):
                for cfg in CONDITIONS[cond]["grid"]:
                    add(cond, d, TUNING_SEED, cfg)
            for ctrl in sorted(set(CE_CONTROL.values())):
                add(ctrl, d, TUNING_SEED, {})
    elif phase == "confirm":
        chosen = load_frozen_selection(root)
        for d in DOMAINS:
            for s in CONFIRMATORY_SEEDS:
                for cond in CONDITIONS:
                    cfg = chosen.get(cond, {}).get("config", {}) if CONDITIONS[cond]["grid"] != [{}] else {}
                    add(cond, d, s, cfg, ["--save-model"] if cond in SAVE_MODEL_CONFIRM else None)
    else:
        raise SystemExit(f"unknown phase {phase}")
    if len({r["run_id"] for r in runs}) != len(runs):
        raise AssertionError("duplicate run ids in plan")
    return runs


def stream(cmd: list[str], log_path: Path, heartbeat: float) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false"))
        sel = selectors.DefaultSelector()
        sel.register(p.stdout, selectors.EVENT_READ)
        last = time.monotonic()
        while True:
            events = sel.select(timeout=heartbeat)
            if events:
                line = p.stdout.readline()
                if not line:
                    break
                fh.write(line)
                fh.flush()
                if "[train]" in line or "Error" in line or "Traceback" in line:
                    print(line, end="", flush=True)
                last = time.monotonic()
            elif time.monotonic() - last >= heartbeat:
                print(f"  ... still running ({log_path.name})", flush=True)
                last = time.monotonic()
            if p.poll() is not None and not events:
                break
        for line in p.stdout:
            fh.write(line)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("plan", "run"))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--phase", required=True, choices=("regression", "tuning", "confirm"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0)
    ap.add_argument("--skip-preflight-check", action="store_true", help="plan tuning without a passed preflight (tests only)")
    args = ap.parse_args()
    check_protocol()
    root = args.root.expanduser().resolve()
    plan_path = root / STAGE1_SUBDIR / "plans" / f"{args.phase}_plan.json"
    if args.action == "plan":
        runs = plan(root, args.phase, args.python, args.skip_preflight_check)
        atomic_write_json(plan_path, {"phase": args.phase, "runs": runs})
        log("plan", f"{args.phase}: {len(runs)} runs -> {plan_path}")
        return 0
    if not plan_path.exists():
        log("plan", f"ERROR: no plan at {plan_path}; run the plan action first")
        return 1
    runs = json.loads(plan_path.read_text(encoding="utf-8"))["runs"]
    if not args.allow_cpu:
        import torch
        if not torch.cuda.is_available():
            log("run", "ERROR: no CUDA GPU; choose a GPU runtime (or --allow-cpu for tests)")
            return 1
    t0, ok, failed = time.monotonic(), 0, 0
    for i, r in enumerate(runs, 1):
        status = "complete" if (Path(r["out_dir"]) / "DONE.json").exists() else "pending"
        log("run", f"[{i}/{len(runs)}] {r['run_id']} ({status}); elapsed {(time.monotonic() - t0) / 60:.1f} min")
        rc = stream(r["command"], Path(r["out_dir"]).parent.parent.parent / "logs" / args.phase / f"{r['run_id']}.log",
                    args.heartbeat_seconds)
        if rc == 0:
            ok += 1
        else:
            failed += 1
            log("run", f"FAILED {r['run_id']} (exit {rc})")
            if not args.keep_going:
                break
    log("run", f"{args.phase}: {ok} ok, {failed} failed, {len(runs) - ok - failed} not attempted; "
               f"{(time.monotonic() - t0) / 60:.1f} min")
    return 0 if failed == 0 and ok == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
