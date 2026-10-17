#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path

from absa_xfer.campaign import CORE_EVALS, LOPO_CONDITIONS
from absa_xfer.hashing import sha256_file
from absa_xfer.io_utils import read_jsonl

FROZEN_BUDGETS = {"restaurant": 1004, "laptop": 419}
FROZEN_MODEL = "FacebookAI/roberta-base"
FROZEN_SEEDS = {13, 37, 73}
HELD_OUT = {
    "Clean+RevTgt+RevNon": "AddDiff",
    "Clean+RevTgt+AddDiff": "RevNon",
    "Clean+RevNon+AddDiff": "RevTgt",
}
PATH_FLAGS = (
    "--train",
    "--dev",
    "--split-manifest",
    "--transformation-manifest",
    "--source-selection",
    "--repo-root",
    "--master-manifest",
    "--out-dir",
)


def _commands(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.startswith("set ")
    ]


def _flag_values(tokens: list[str], flag: str) -> list[str]:
    out: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if token == flag:
            out.append(tokens[idx + 1])
    return out


def _load_selection(path: Path) -> tuple[list[str], dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Selection file must be a JSON object: {path}")
    ids = obj.get("selected_ids")
    if ids is None:
        ids = obj.get("selected_source_ids")
    if not isinstance(ids, list):
        raise RuntimeError(
            f"Selection file lacks selected_ids/selected_source_ids list: {path}"
        )
    return [str(x) for x in ids], obj


def _expected_eval_paths(root: Path, domain: str) -> dict[str, Path]:
    base = root / "data_processed" / domain
    return {
        "Clean": base / "test.jsonl",
        "RevTgt": base / "evaluation" / "RevTgt.jsonl",
        "RevNon": base / "evaluation" / "RevNon.jsonl",
        "AddDiff": base / "evaluation" / "AddDiff.jsonl",
    }


def _family_quota_expected(domain: str, condition: str) -> dict[str, int]:
    B = FROZEN_BUDGETS[domain]
    a, b = condition.removeprefix("Clean+").split("+")
    half = B // 2
    counts = {a: half, b: half}
    if B % 2:
        extra = {
            "Clean+RevTgt+RevNon": "RevTgt",
            "Clean+RevTgt+AddDiff": "AddDiff",
            "Clean+RevNon+AddDiff": "RevNon",
        }[condition]
        counts[extra] += 1
    return counts


def _validate_materialized_domain(root: Path, domain: str, problems: list[dict]) -> dict:
    B = FROZEN_BUDGETS[domain]
    selection_path = root / "audits" / "eligibility" / domain / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"
    manifest_path = root / "transformations" / "materialized" / domain / "LOPO_MANIFEST.json"
    result = {
        "domain": domain,
        "budget": B,
        "selection_path": str(selection_path),
        "manifest_path": str(manifest_path),
        "conditions": {},
    }

    if not selection_path.is_file():
        problems.append({"scope": domain, "problem": "missing frozen source selection", "path": str(selection_path)})
        return result
    if not manifest_path.is_file():
        problems.append({"scope": domain, "problem": "missing LOPO manifest", "path": str(manifest_path)})
        return result

    try:
        selected, selection_obj = _load_selection(selection_path)
    except Exception as exc:
        problems.append({"scope": domain, "problem": f"invalid source selection: {exc}"})
        return result

    if len(selected) != B or len(set(selected)) != B:
        problems.append({
            "scope": domain,
            "problem": "frozen selected-source cardinality mismatch",
            "expected": B,
            "found": len(selected),
            "unique": len(set(selected)),
        })
    declared_budget = selection_obj.get("budget")
    if declared_budget is not None and int(declared_budget) != B:
        problems.append({
            "scope": domain,
            "problem": "selection-declared budget differs from frozen budget",
            "expected": B,
            "found": declared_budget,
        })

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("total_budget", -1)) != B:
        problems.append({"scope": domain, "problem": "LOPO manifest budget mismatch", "expected": B, "found": manifest.get("total_budget")})
    if manifest.get("source_selection_sha256") != sha256_file(selection_path):
        problems.append({"scope": domain, "problem": "LOPO manifest source-selection hash mismatch"})
    if manifest.get("source_mode") != "reuse_frozen_single_family_materializations":
        problems.append({
            "scope": domain,
            "problem": "LOPO manifest does not certify frozen single-family reuse",
            "found": manifest.get("source_mode"),
        })

    core_manifest_path = root / "manifests" / domain / "transformation_manifest.json"
    if not core_manifest_path.is_file():
        problems.append({"scope": domain, "problem": "missing core transformation manifest", "path": str(core_manifest_path)})
        core_manifest = {}
    else:
        core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("core_transformation_manifest_sha256") != sha256_file(core_manifest_path):
            problems.append({"scope": domain, "problem": "LOPO core transformation-manifest hash mismatch"})

    frozen_family_sha = manifest.get("frozen_family_sha256") or {}
    core_conditions = core_manifest.get("conditions") or {}
    for family in ("RevTgt", "RevNon", "AddDiff"):
        family_path = root / "transformations" / "materialized" / domain / f"Clean+{family}" / "train.jsonl"
        if not family_path.is_file():
            problems.append({"scope": domain, "problem": f"missing frozen core family file {family}", "path": str(family_path)})
            continue
        actual_sha = sha256_file(family_path)
        if frozen_family_sha.get(family) != actual_sha:
            problems.append({"scope": domain, "problem": f"LOPO frozen-family hash mismatch for {family}"})
        core_entry = core_conditions.get(f"Clean+{family}")
        if not isinstance(core_entry, dict) or core_entry.get("sha256") != actual_sha:
            problems.append({"scope": domain, "problem": f"core transformation manifest hash mismatch for Clean+{family}"})

    # Protected firewall: all official protected evaluation sentences.
    protected_sentences: set[str] = set()
    eval_paths = _expected_eval_paths(root, domain)
    for name, path in eval_paths.items():
        if not path.is_file():
            problems.append({"scope": domain, "problem": f"missing protected eval file {name}", "path": str(path)})
            continue
        for row in read_jsonl(path):
            if "sentence" in row:
                protected_sentences.add(str(row["sentence"]))

    for condition in LOPO_CONDITIONS:
        path = root / "transformations" / "materialized" / domain / condition / "train.jsonl"
        info = {
            "path": str(path),
            "held_out_family": HELD_OUT[condition],
        }
        result["conditions"][condition] = info
        if not path.is_file():
            problems.append({"scope": domain, "condition": condition, "problem": "missing LOPO train file", "path": str(path)})
            continue

        rows = read_jsonl(path)
        aug = [r for r in rows if bool(r.get("is_augmented"))]
        clean = [r for r in rows if not bool(r.get("is_augmented"))]
        aug_ids = [str(r.get("source_instance_id")) for r in aug]
        fam_counts = Counter(str(r.get("perturbation_family")) for r in aug)
        expected_fam_counts = _family_quota_expected(domain, condition)
        overlap = sorted({str(r.get("sentence")) for r in rows} & protected_sentences)

        info.update({
            "sha256": sha256_file(path),
            "n_rows": len(rows),
            "n_clean": len(clean),
            "n_augmented": len(aug),
            "unique_augmented_sources": len(set(aug_ids)),
            "family_counts": dict(fam_counts),
            "protected_sentence_overlap_count": len(overlap),
        })

        if len(aug) != B:
            problems.append({"scope": domain, "condition": condition, "problem": "augmentation budget mismatch", "expected": B, "found": len(aug)})
        if Counter(aug_ids) != Counter(selected):
            problems.append({"scope": domain, "condition": condition, "problem": "augmented source composition differs from frozen selected pool"})
        if dict(fam_counts) != expected_fam_counts:
            problems.append({"scope": domain, "condition": condition, "problem": "family quota mismatch", "expected": expected_fam_counts, "found": dict(fam_counts)})
        if HELD_OUT[condition] in fam_counts:
            problems.append({"scope": domain, "condition": condition, "problem": "held-out family leaked into training", "held_out": HELD_OUT[condition]})
        if overlap:
            problems.append({"scope": domain, "condition": condition, "problem": "protected-test sentence overlap", "count": len(overlap), "examples": overlap[:5]})

        manifest_entry = (manifest.get("conditions") or {}).get(condition)
        if not isinstance(manifest_entry, dict):
            problems.append({"scope": domain, "condition": condition, "problem": "condition missing from LOPO manifest"})
        else:
            if manifest_entry.get("sha256") != sha256_file(path):
                problems.append({"scope": domain, "condition": condition, "problem": "LOPO manifest train hash mismatch"})
            if int(manifest_entry.get("n_augmented", -1)) != B:
                problems.append({"scope": domain, "condition": condition, "problem": "LOPO manifest augmented count mismatch"})
            if manifest_entry.get("held_out_family") != HELD_OUT[condition]:
                problems.append({"scope": domain, "condition": condition, "problem": "LOPO manifest held-out family mismatch"})

    return result


def validate(root: Path, campaign_dir: Path) -> dict:
    problems: list[dict] = []
    materialization = {
        domain: _validate_materialized_domain(root, domain, problems)
        for domain in ("restaurant", "laptop")
    }

    plan_path = campaign_dir / "lopo_plan.csv"
    commands_path = campaign_dir / "lopo_commands.sh"
    if not plan_path.is_file():
        problems.append({"scope": "campaign", "problem": "missing lopo_plan.csv", "path": str(plan_path)})
        rows = []
    else:
        with plan_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    commands = _commands(commands_path) if commands_path.is_file() else []
    if not commands_path.is_file():
        problems.append({"scope": "campaign", "problem": "missing lopo_commands.sh", "path": str(commands_path)})

    if len(rows) != 18:
        problems.append({"scope": "plan", "problem": f"expected 18 rows, found {len(rows)}"})
    if len(commands) != 18:
        problems.append({"scope": "commands", "problem": f"expected 18 commands, found {len(commands)}"})

    run_ids = [r.get("run_id", "") for r in rows]
    if len(set(run_ids)) != 18 or any(not x for x in run_ids):
        problems.append({"scope": "plan", "problem": "run IDs are not 18 nonempty unique values"})

    expected_coverage = {
        (domain, condition, seed)
        for domain in ("restaurant", "laptop")
        for condition in LOPO_CONDITIONS
        for seed in FROZEN_SEEDS
    }
    found_coverage = {
        (r.get("dataset"), r.get("training_condition"), int(r.get("seed", -1)))
        for r in rows
    }
    if found_coverage != expected_coverage:
        problems.append({
            "scope": "plan",
            "problem": "LOPO coverage mismatch",
            "missing": sorted(expected_coverage - found_coverage),
            "extra": sorted(found_coverage - expected_coverage),
        })
    if {r.get("model") for r in rows} != {FROZEN_MODEL}:
        problems.append({"scope": "plan", "problem": "LOPO plan uses model(s) other than frozen RoBERTa-base"})
    if any(not r.get("model_revision") for r in rows):
        problems.append({"scope": "plan", "problem": "one or more LOPO rows lack immutable model revision"})

    signatures = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in commands]
    if len(signatures) != len(set(signatures)):
        problems.append({"scope": "commands", "problem": "duplicate command signatures"})

    for idx, line in enumerate(commands, 1):
        tokens = shlex.split(line)
        if not tokens or tokens[0] != "/usr/bin/python3":
            problems.append({"command": idx, "problem": "interpreter is not /usr/bin/python3"})
            continue

        def once(flag: str) -> str | None:
            vals = _flag_values(tokens, flag)
            if len(vals) != 1:
                problems.append({"command": idx, "problem": f"{flag} expected once, found {len(vals)}"})
                return None
            return vals[0]

        domain = once("--dataset")
        condition = once("--condition")
        seed_raw = once("--seed")
        model = once("--model")
        model_rev = once("--model-revision")
        tokenizer_rev = once("--tokenizer-revision")
        pair_order = once("--pair-order")
        if domain not in {"restaurant", "laptop"}:
            problems.append({"command": idx, "problem": f"invalid domain {domain}"})
            continue
        if condition not in LOPO_CONDITIONS:
            problems.append({"command": idx, "problem": f"invalid LOPO condition {condition}"})
            continue
        try:
            seed = int(seed_raw) if seed_raw is not None else -1
        except ValueError:
            seed = -1
        if seed not in FROZEN_SEEDS:
            problems.append({"command": idx, "problem": f"invalid LOPO seed {seed_raw}"})
        if model != FROZEN_MODEL:
            problems.append({"command": idx, "problem": f"invalid LOPO model {model}"})
        if not model_rev or model_rev != tokenizer_rev:
            problems.append({"command": idx, "problem": "model/tokenizer revision missing or inconsistent"})
        if pair_order != "sentence_first":
            problems.append({"command": idx, "problem": f"pair order is not sentence_first: {pair_order}"})

        expected_paths = {
            "--train": root / "transformations" / "materialized" / domain / condition / "train.jsonl",
            "--dev": root / "data_processed" / domain / "dev.jsonl",
            "--split-manifest": root / "manifests" / domain / "split_manifest.json",
            "--transformation-manifest": root / "transformations" / "materialized" / domain / "LOPO_MANIFEST.json",
            "--source-selection": root / "audits" / "eligibility" / domain / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json",
            "--repo-root": root / "code",
            "--master-manifest": root / "manifests" / "master_run_manifest.jsonl",
            "--out-dir": root / "runs" / "lopo" / next((r["run_id"] for r in rows if r.get("dataset") == domain and r.get("training_condition") == condition and int(r.get("seed", -1)) == seed), "__MISSING__"),
        }
        for flag, expected in expected_paths.items():
            got = once(flag)
            if got != str(expected):
                problems.append({"command": idx, "problem": f"{flag} path mismatch", "expected": str(expected), "found": got})
                continue
            p = Path(got)
            # out-dir need not exist before launch; master manifest may be created by append.
            if flag in {"--out-dir", "--master-manifest"}:
                continue
            ok = p.is_dir() if flag == "--repo-root" else p.is_file()
            if not ok:
                problems.append({"command": idx, "problem": f"missing path for {flag}", "path": str(p)})

        eval_items = _flag_values(tokens, "--eval")
        eval_map: dict[str, str] = {}
        for item in eval_items:
            if "=" not in item:
                problems.append({"command": idx, "problem": f"malformed --eval {item!r}"})
                continue
            name, value = item.split("=", 1)
            eval_map[name] = value
        if set(eval_map) != set(CORE_EVALS):
            problems.append({"command": idx, "problem": f"eval names mismatch: {sorted(eval_map)}"})
        for name, expected in _expected_eval_paths(root, domain).items():
            got = eval_map.get(name)
            if got != str(expected):
                problems.append({"command": idx, "problem": f"{name} eval path mismatch", "expected": str(expected), "found": got})
            elif not expected.is_file():
                problems.append({"command": idx, "problem": f"missing eval file {name}", "path": str(expected)})

    audit = {
        "root": str(root),
        "campaign_dir": str(campaign_dir),
        "frozen_budgets": FROZEN_BUDGETS,
        "n_plan_rows": len(rows),
        "n_commands": len(commands),
        "unique_run_ids": len(set(run_ids)),
        "unique_command_signatures": len(set(signatures)),
        "materialization": materialization,
        "problems": problems,
        "status": "READY_NOT_LAUNCHED" if not problems else "BLOCKED",
    }
    audit_path = root / "audits" / "LOPO_STAGE_PREFLIGHT.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if problems:
        raise RuntimeError(
            f"LOPO stage preflight BLOCKED with {len(problems)} problem(s). "
            f"See {audit_path}. First: {problems[:3]}"
        )
    return audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--campaign-dir", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    campaign_dir = Path(args.campaign_dir) if args.campaign_dir else root / "campaigns" / "lopo_minimal"
    audit = validate(root, campaign_dir)
    print(json.dumps(audit, indent=2))
    print("\nLOPO STAGE PREFLIGHT: READY_NOT_LAUNCHED")


if __name__ == "__main__":
    main()
