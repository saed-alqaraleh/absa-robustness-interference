#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path

CORE_EVALS = ("Clean", "RevTgt", "RevNon", "AddDiff")
PATH_FLAGS = ("--train", "--dev", "--split-manifest", "--transformation-manifest", "--source-selection", "--repo-root")


def _commands(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("set ")
    ]


def _flag_values(tokens: list[str], flag: str) -> list[str]:
    out: list[str] = []
    for i, token in enumerate(tokens[:-1]):
        if token == flag:
            out.append(tokens[i + 1])
    return out


def expected_eval_paths(root: Path, domain: str) -> dict[str, Path]:
    base = root / "data_processed" / domain
    return {
        "Clean": base / "test.jsonl",
        "RevTgt": base / "evaluation" / "RevTgt.jsonl",
        "RevNon": base / "evaluation" / "RevNon.jsonl",
        "AddDiff": base / "evaluation" / "AddDiff.jsonl",
    }


def validate(root: Path, campaign_dir: Path) -> dict:
    plan_path = campaign_dir / "minimal_plan.csv"
    commands_path = campaign_dir / "minimal_commands.sh"
    if not plan_path.exists() or not commands_path.exists():
        raise RuntimeError(f"Campaign artifacts missing under {campaign_dir}")

    with plan_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    commands = _commands(commands_path)

    problems: list[dict] = []
    if len(rows) != 36:
        problems.append({"scope": "plan", "problem": f"expected 36 rows, found {len(rows)}"})
    run_ids = [r.get("run_id", "") for r in rows]
    if len(run_ids) != 36 or len(set(run_ids)) != 36 or any(not x for x in run_ids):
        problems.append({"scope": "plan", "problem": "run IDs are not 36 nonempty unique values"})
    if len(commands) != 36:
        problems.append({"scope": "commands", "problem": f"expected 36 commands, found {len(commands)}"})

    signatures = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in commands]
    if len(signatures) != len(set(signatures)):
        problems.append({"scope": "commands", "problem": "duplicate command signatures"})

    checked_paths: set[str] = set()
    for idx, line in enumerate(commands, 1):
        tokens = shlex.split(line)
        if not tokens or tokens[0] != "/usr/bin/python3":
            problems.append({"command": idx, "problem": "interpreter is not /usr/bin/python3"})
            continue
        datasets = _flag_values(tokens, "--dataset")
        if len(datasets) != 1 or datasets[0] not in {"restaurant", "laptop"}:
            problems.append({"command": idx, "problem": f"invalid --dataset values: {datasets}"})
            continue
        domain = datasets[0]

        eval_items = _flag_values(tokens, "--eval")
        eval_map: dict[str, str] = {}
        for item in eval_items:
            if "=" not in item:
                problems.append({"command": idx, "problem": f"malformed --eval {item!r}"})
                continue
            name, value = item.split("=", 1)
            if name in eval_map:
                problems.append({"command": idx, "problem": f"duplicate eval name {name}"})
            eval_map[name] = value
        if set(eval_map) != set(CORE_EVALS):
            problems.append({"command": idx, "problem": f"eval names are {sorted(eval_map)}, expected {list(CORE_EVALS)}"})

        expected = expected_eval_paths(root, domain)
        for name, exp in expected.items():
            got = eval_map.get(name)
            if got != str(exp):
                problems.append({"command": idx, "problem": f"{name} path mismatch", "expected": str(exp), "got": got})
            if got:
                p = Path(got)
                checked_paths.add(str(p))
                if not p.is_file():
                    problems.append({"command": idx, "problem": f"missing eval file {name}", "path": str(p)})

        for flag in PATH_FLAGS:
            vals = _flag_values(tokens, flag)
            if len(vals) != 1:
                problems.append({"command": idx, "problem": f"{flag} expected once, found {len(vals)}"})
                continue
            p = Path(vals[0])
            checked_paths.add(str(p))
            ok = p.is_dir() if flag == "--repo-root" else p.is_file()
            if not ok:
                problems.append({"command": idx, "problem": f"missing path for {flag}", "path": str(p)})

    audit = {
        "root": str(root),
        "campaign_dir": str(campaign_dir),
        "n_plan_rows": len(rows),
        "n_commands": len(commands),
        "unique_run_ids": len(set(run_ids)),
        "unique_command_signatures": len(set(signatures)),
        "checked_unique_input_paths": len(checked_paths),
        "problems": problems,
        "status": "READY_TO_LAUNCH" if not problems else "BLOCKED",
    }
    audit_path = root / "audits" / "MINIMAL_CAMPAIGN_FILE_PREFLIGHT.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if problems:
        raise RuntimeError(f"Campaign file preflight BLOCKED with {len(problems)} problem(s). See {audit_path}. First: {problems[:3]}")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--campaign-dir", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    campaign_dir = Path(args.campaign_dir) if args.campaign_dir else root / "campaigns" / "minimal_decisive"
    audit = validate(root, campaign_dir)
    print(json.dumps(audit, indent=2))
    print("\nCAMPAIGN FILE PREFLIGHT: READY_TO_LAUNCH")


if __name__ == "__main__":
    main()
