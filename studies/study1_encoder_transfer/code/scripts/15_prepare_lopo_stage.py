#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from absa_xfer.hashing import sha256_file

FROZEN_BUDGETS = {"restaurant": 1004, "laptop": 419}
FROZEN_SEEDS = {13, 37, 73}
FROZEN_MODEL = "FacebookAI/roberta-base"
CORE_CONDITIONS = {
    "Clean-Natural", "Clean-Matched", "Clean+RevTgt",
    "Clean+RevNon", "Clean+AddDiff", "Clean+Mixed",
}
CORE_EVALS = {"Clean", "RevTgt", "RevNon", "AddDiff"}
CORE_METRICS = {"accuracy", "macro_f1"}


def validate_continue_decision(decision: str | None) -> str:
    if decision != "CONTINUE":
        raise RuntimeError(
            "LOPO preparation is protocol-gated and requires the explicit CONTINUE decision"
        )
    return decision


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _audit_minimal_results(root: Path) -> dict:
    path = root / "results" / "minimal_decisive_audit" / "results_master_minimal.csv"
    if not path.is_file():
        raise RuntimeError(
            f"Minimal decisive evidence is missing: {path}. "
            "LOPO preparation remains blocked."
        )
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    expected_rows = 36 * 4 * 2
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Minimal evidence row count mismatch: {len(rows)} != {expected_rows}"
        )

    run_keys = {
        (
            r["dataset"], r["training_condition"], int(r["seed"]),
            r["model"], r["model_revision"],
        )
        for r in rows
    }
    if len(run_keys) != 36:
        raise RuntimeError(f"Minimal evidence does not contain 36 unique runs: {len(run_keys)}")

    expected_run_keys = {
        (domain, condition, seed)
        for domain in ("restaurant", "laptop")
        for condition in CORE_CONDITIONS
        for seed in FROZEN_SEEDS
    }
    found_run_keys = {(d, c, s) for d, c, s, _, _ in run_keys}
    if found_run_keys != expected_run_keys:
        raise RuntimeError("Minimal evidence coverage differs from the frozen 2x6x3 design")
    if {m for _, _, _, m, _ in run_keys} != {FROZEN_MODEL}:
        raise RuntimeError("Minimal evidence contains a model other than frozen RoBERTa-base")
    revisions = {rev for _, _, _, _, rev in run_keys}
    if len(revisions) != 1 or not next(iter(revisions)):
        raise RuntimeError(f"Minimal evidence model revision is not uniquely pinned: {revisions}")

    evals = {r["evaluation_condition"] for r in rows}
    metrics = {r["metric"] for r in rows}
    if evals != CORE_EVALS or metrics != CORE_METRICS:
        raise RuntimeError(
            f"Minimal evidence endpoint coverage mismatch: evals={sorted(evals)}, metrics={sorted(metrics)}"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_rows": len(rows),
        "n_unique_runs": len(run_keys),
        "model": FROZEN_MODEL,
        "model_revision": next(iter(revisions)),
    }



def build_materialize_command(
    root: Path, domain: str, budget: int, python_executable: str = "/usr/bin/python3"
) -> list[str]:
    train = root / "data_processed" / domain / "train.jsonl"
    selected = root / "audits" / "eligibility" / domain / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"
    out = root / "transformations" / "materialized" / domain
    core_manifest = root / "manifests" / domain / "transformation_manifest.json"
    cmd = [
        python_executable,
        "scripts/12_materialize_lopo.py",
        "--train", str(train),
        "--selected", str(selected),
        "--core-manifest", str(core_manifest),
        "--out-dir", str(out),
        "--seed", "20260913",
        "--expected-budget", str(budget),
    ]
    for family in ("RevTgt", "RevNon", "AddDiff"):
        family_path = out / f"Clean+{family}" / "train.jsonl"
        cmd.extend(["--family-train", f"{family}={family_path}"])
    for eval_path in (
        root / "data_processed" / domain / "test.jsonl",
        root / "data_processed" / domain / "evaluation" / "RevTgt.jsonl",
        root / "data_processed" / domain / "evaluation" / "RevNon.jsonl",
        root / "data_processed" / domain / "evaluation" / "AddDiff.jsonl",
    ):
        cmd.extend(["--protected-eval", str(eval_path)])
    return cmd

def prepare(root: Path, decision: str, python_executable: str = "/usr/bin/python3") -> dict:
    validate_continue_decision(decision)
    code = root / "code"
    if not code.is_dir():
        raise RuntimeError(f"Project code directory missing: {code}")

    minimal_evidence = _audit_minimal_results(root)
    step9 = root / "audits" / "STEP9_COMPLETION.json"
    if not step9.is_file():
        raise RuntimeError(f"Step-9 completion audit missing: {step9}")
    step9_obj = json.loads(step9.read_text(encoding="utf-8"))
    if step9_obj.get("status") != "PASS_READY_NOT_LAUNCHED":
        raise RuntimeError(
            f"Step-9 protected-evaluation gate is not PASS_READY_NOT_LAUNCHED: "
            f"{step9_obj.get('status')}"
        )

    pins = root / "manifests" / "model_pins.json"
    if not pins.is_file():
        raise RuntimeError(f"Frozen model-pin manifest missing: {pins}")
    pin_obj = json.loads(pins.read_text(encoding="utf-8"))
    pin_map = pin_obj.get("pins", pin_obj)
    pinned_rev = (pin_map.get(FROZEN_MODEL) or {}).get("revision")
    if not pinned_rev:
        raise RuntimeError(f"Frozen RoBERTa revision missing from {pins}")
    if pinned_rev != minimal_evidence["model_revision"]:
        raise RuntimeError(
            "LOPO model pin differs from the completed minimal decisive campaign: "
            f"{pinned_rev} != {minimal_evidence['model_revision']}"
        )

    gate = {
        "decision": decision,
        "protocol_transition": "minimal_decisive_to_leave_one_perturbation_out",
        "minimal_evidence": minimal_evidence,
        "step9_completion_sha256": sha256_file(step9),
        "model_pins_sha256": sha256_file(pins),
        "frozen_model": FROZEN_MODEL,
        "frozen_model_revision": pinned_rev,
        "frozen_seeds": sorted(FROZEN_SEEDS),
        "frozen_budgets": FROZEN_BUDGETS,
    }
    gate_path = root / "audits" / "LOPO_CONTINUE_GATE.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    for domain, budget in FROZEN_BUDGETS.items():
        _run(
            build_materialize_command(root, domain, budget, python_executable),
            cwd=code,
        )

    campaign_dir = root / "campaigns" / "lopo_minimal"
    _run([
        python_executable,
        "scripts/11_build_campaign.py",
        "--config", "configs/base.yaml",
        "--stage", "lopo",
        "--out-dir", str(campaign_dir),
        "--model-pins", str(pins),
    ], cwd=code)

    _run([
        python_executable,
        "scripts/16_validate_lopo_stage.py",
        "--root", str(root),
        "--campaign-dir", str(campaign_dir),
    ], cwd=code)

    plan = campaign_dir / "lopo_plan.csv"
    commands = campaign_dir / "lopo_commands.sh"
    completion = {
        "status": "READY_NOT_LAUNCHED",
        "decision_gate": str(gate_path),
        "decision_gate_sha256": sha256_file(gate_path),
        "restaurant_lopo_manifest_sha256": sha256_file(root / "transformations" / "materialized" / "restaurant" / "LOPO_MANIFEST.json"),
        "laptop_lopo_manifest_sha256": sha256_file(root / "transformations" / "materialized" / "laptop" / "LOPO_MANIFEST.json"),
        "campaign_plan": str(plan),
        "campaign_plan_sha256": sha256_file(plan),
        "campaign_commands": str(commands),
        "campaign_commands_sha256": sha256_file(commands),
        "n_runs": 18,
        "launch_performed": False,
    }
    completion_path = root / "audits" / "LOPO_PREPARATION_COMPLETION.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    return completion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--decision", required=True)
    args = ap.parse_args()
    result = prepare(Path(args.root), args.decision)
    print("\n" + "=" * 108)
    print("LOPO PREPARATION COMPLETE — READY_NOT_LAUNCHED")
    print("=" * 108)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
