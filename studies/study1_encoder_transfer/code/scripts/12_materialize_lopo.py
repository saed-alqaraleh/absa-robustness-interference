#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from absa_xfer.hashing import hash_object, sha256_file
from absa_xfer.io_utils import atomic_write_json, read_jsonl, write_jsonl
from absa_xfer.materialize import materialize_lopo_conditions, materialize_lopo_from_frozen_rows
from absa_xfer.schemas import AspectExample, OpinionRecord, OpinionSpan

LOPO_CONDITIONS = (
    "Clean+RevTgt+RevNon",
    "Clean+RevTgt+AddDiff",
    "Clean+RevNon+AddDiff",
)
HELD_OUT = {
    "Clean+RevTgt+RevNon": "AddDiff",
    "Clean+RevTgt+AddDiff": "RevNon",
    "Clean+RevNon+AddDiff": "RevTgt",
}
ODD_EXTRA = {
    "Clean+RevTgt+RevNon": "RevTgt",
    "Clean+RevTgt+AddDiff": "AddDiff",
    "Clean+RevNon+AddDiff": "RevNon",
}


def _load_examples(path: Path) -> list[AspectExample]:
    return [
        AspectExample(**{k: row[k] for k in AspectExample.__dataclass_fields__})
        for row in read_jsonl(path)
    ]


def _load_opinions(path: Path) -> dict[str, OpinionRecord]:
    out: dict[str, OpinionRecord] = {}
    for row in read_jsonl(path):
        spans = tuple(OpinionSpan(**span) for span in row["spans"])
        rec = OpinionRecord(
            row["source_instance_id"],
            row["cluster_id"],
            spans,
            row["matched_by"],
            row["provenance"],
        )
        if rec.source_instance_id in out:
            raise RuntimeError(f"Duplicate opinion record: {rec.source_instance_id}")
        out[rec.source_instance_id] = rec
    return out


def _load_selection(path: Path) -> tuple[list[str], dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        ids = obj.get("selected_ids")
        if ids is None:
            ids = obj.get("selected_source_ids")
        if not isinstance(ids, list):
            raise RuntimeError(
                f"Selection JSON lacks selected_ids/selected_source_ids list: {path}"
            )
        return [str(x) for x in ids], obj
    if isinstance(obj, list):
        return [str(x) for x in obj], {}
    raise RuntimeError(f"Unsupported source-selection format: {path}")


def _family_quota_expected(B: int, condition: str) -> dict[str, int]:
    fa, fb = condition.removeprefix("Clean+").split("+")
    counts = {fa: B // 2, fb: B // 2}
    if B % 2:
        counts[ODD_EXTRA[condition]] += 1
    return counts


def materialize_domain(
    train_path: str | Path,
    opinions_path: str | Path,
    selected_path: str | Path,
    out_dir: str | Path,
    *,
    seed: int,
    expected_budget: int,
    protected_eval_paths: list[str | Path],
) -> dict:
    """Materialize and freeze one domain's three LOPO training conditions.

    This function is intentionally fail-closed.  It validates the frozen source
    budget, exact source composition, pairwise family quotas, held-out-family
    absence, protected-test firewall, and all input/output hashes before writing
    the final LOPO manifest.
    """
    train_path = Path(train_path)
    opinions_path = Path(opinions_path)
    selected_path = Path(selected_path)
    out_dir = Path(out_dir)
    manifest_path = out_dir / "LOPO_MANIFEST.json"

    for path, label in (
        (train_path, "train"),
        (opinions_path, "opinions"),
        (selected_path, "source selection"),
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing LOPO {label} input: {path}")

    selected_ids, selection_obj = _load_selection(selected_path)
    if len(selected_ids) != expected_budget:
        raise RuntimeError(
            f"Frozen LOPO budget mismatch: expected {expected_budget}, "
            f"selection contains {len(selected_ids)} source IDs"
        )
    if len(set(selected_ids)) != expected_budget:
        raise RuntimeError("Frozen LOPO source selection contains duplicate IDs")
    if selection_obj.get("budget") is not None and int(selection_obj["budget"]) != expected_budget:
        raise RuntimeError(
            f"Frozen LOPO selection-declared budget mismatch: "
            f"{selection_obj['budget']} != {expected_budget}"
        )

    train_sha = sha256_file(train_path)
    opinions_sha = sha256_file(opinions_path)
    if selection_obj.get("input_train_sha256") and selection_obj["input_train_sha256"] != train_sha:
        raise RuntimeError("Frozen source selection was built from a different train file")
    if selection_obj.get("input_opinions_sha256") and selection_obj["input_opinions_sha256"] != opinions_sha:
        raise RuntimeError("Frozen source selection was built from a different opinion file")

    examples = _load_examples(train_path)
    if not examples:
        raise RuntimeError("LOPO train set is empty")
    domain = examples[0].domain
    if any(x.domain != domain for x in examples):
        raise RuntimeError("LOPO train file contains mixed domains")
    opinions = _load_opinions(opinions_path)

    train_ids = {x.source_instance_id for x in examples}
    missing_selected = sorted(set(selected_ids) - train_ids)
    if missing_selected:
        raise RuntimeError(
            f"Frozen selected sources missing from train file: {missing_selected[:10]}"
        )

    # Remove only the final manifest before generation.  If anything fails below,
    # stale LOPO train files cannot be authorized because the manifest is absent.
    if manifest_path.exists():
        manifest_path.unlink()

    conditions = materialize_lopo_conditions(
        examples,
        opinions,
        selected_ids,
        seed,
        total_budget=expected_budget,
    )
    if set(conditions) != set(LOPO_CONDITIONS):
        raise RuntimeError(
            f"Unexpected LOPO condition set: {sorted(conditions)}"
        )

    protected_sentences: set[str] = set()
    protected_hashes: dict[str, str] = {}
    for raw_path in protected_eval_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"Missing protected evaluation file: {path}")
        protected_hashes[str(path)] = sha256_file(path)
        for row in read_jsonl(path):
            if "sentence" not in row:
                raise RuntimeError(f"Protected evaluation row lacks sentence: {path}")
            protected_sentences.add(str(row["sentence"]))

    # Firewall first: no clean or generated training sentence may equal a protected
    # evaluation sentence.  This mirrors the protected-test discipline of Step 9.
    firewall_overlaps: dict[str, list[str]] = {}
    for condition, rows in conditions.items():
        overlap = sorted({str(row["sentence"]) for row in rows} & protected_sentences)
        if overlap:
            firewall_overlaps[condition] = overlap
    if firewall_overlaps:
        first_condition = sorted(firewall_overlaps)[0]
        raise RuntimeError(
            f"Protected-test sentence overlap in LOPO condition {first_condition}: "
            f"{firewall_overlaps[first_condition][:5]}"
        )

    # Full scientific-contract validation before committing any new train file.
    validated: dict[str, dict] = {}
    selected_counter = Counter(selected_ids)
    for condition, rows in conditions.items():
        aug = [row for row in rows if bool(row.get("is_augmented"))]
        aug_ids = [str(row.get("source_instance_id")) for row in aug]
        family_counts = Counter(str(row.get("perturbation_family")) for row in aug)
        expected_family_counts = _family_quota_expected(expected_budget, condition)
        held_out = HELD_OUT[condition]

        if len(aug) != expected_budget:
            raise RuntimeError(
                f"LOPO augmentation budget mismatch for {condition}: "
                f"{len(aug)} != {expected_budget}"
            )
        if Counter(aug_ids) != selected_counter:
            raise RuntimeError(
                f"LOPO source composition mismatch for {condition}; each frozen "
                "selected source must appear exactly once"
            )
        if dict(family_counts) != expected_family_counts:
            raise RuntimeError(
                f"LOPO family quota mismatch for {condition}: "
                f"{dict(family_counts)} != {expected_family_counts}"
            )
        if held_out in family_counts:
            raise RuntimeError(
                f"LOPO held-out family leaked into {condition}: {held_out}"
            )
        if len(rows) != len(examples) + expected_budget:
            raise RuntimeError(
                f"LOPO total row count mismatch for {condition}: "
                f"{len(rows)} != {len(examples) + expected_budget}"
            )

        validated[condition] = {
            "n_rows": len(rows),
            "n_clean": len(rows) - len(aug),
            "n_augmented": len(aug),
            "unique_augmented_sources": len(set(aug_ids)),
            "family_counts": dict(family_counts),
            "held_out_family": held_out,
            "protected_sentence_overlap_count": 0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_conditions: dict[str, dict] = {}
    for condition in LOPO_CONDITIONS:
        path = out_dir / condition / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, conditions[condition])
        manifest_conditions[condition] = {
            **validated[condition],
            "path": str(path),
            "sha256": sha256_file(path),
        }

    manifest = {
        "protocol_stage": "leave_one_perturbation_out_minimal",
        "implementation": "ARTS-inspired deterministic LOPO v2",
        "domain": domain,
        "seed": seed,
        "total_budget": expected_budget,
        "source_policy": "frozen_common_pool_exactly_once_per_condition",
        "odd_budget_policy": "rotate_single_extra_slot_symmetrically_across_families",
        "source_selection_path": str(selected_path),
        "source_selection_sha256": sha256_file(selected_path),
        "selected_ids_hash": hash_object(selected_ids),
        "train_path": str(train_path),
        "train_sha256": train_sha,
        "opinions_path": str(opinions_path),
        "opinions_sha256": opinions_sha,
        "protected_eval_sha256": protected_hashes,
        "conditions": manifest_conditions,
    }
    manifest["manifest_hash"] = hash_object(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest



def materialize_domain_from_frozen(
    train_path: str | Path,
    selected_path: str | Path,
    core_manifest_path: str | Path,
    family_paths: dict[str, str | Path],
    out_dir: str | Path,
    *,
    seed: int,
    expected_budget: int,
    protected_eval_paths: list[str | Path],
) -> dict:
    """Compose LOPO rows exclusively from frozen single-family materializations.

    This is the production path for the scientific LOPO stage. No transformation
    generator is called here. The function verifies the core transformation
    manifest hashes, exact selected-source coverage for each family, held-out
    family absence, protected-test firewall, and writes a new LOPO manifest last.
    """
    train_path = Path(train_path)
    selected_path = Path(selected_path)
    core_manifest_path = Path(core_manifest_path)
    out_dir = Path(out_dir)
    manifest_path = out_dir / "LOPO_MANIFEST.json"

    for path, label in (
        (train_path, "train"),
        (selected_path, "source selection"),
        (core_manifest_path, "core transformation manifest"),
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing LOPO {label} input: {path}")

    selected_ids, selection_obj = _load_selection(selected_path)
    if len(selected_ids) != expected_budget:
        raise RuntimeError(
            f"Frozen LOPO budget mismatch: expected {expected_budget}, "
            f"selection contains {len(selected_ids)} source IDs"
        )
    if len(set(selected_ids)) != expected_budget:
        raise RuntimeError("Frozen LOPO source selection contains duplicate IDs")
    if selection_obj.get("budget") is not None and int(selection_obj["budget"]) != expected_budget:
        raise RuntimeError(
            f"Frozen LOPO selection-declared budget mismatch: "
            f"{selection_obj['budget']} != {expected_budget}"
        )

    examples = _load_examples(train_path)
    if not examples:
        raise RuntimeError("LOPO train set is empty")
    domain = examples[0].domain
    if any(x.domain != domain for x in examples):
        raise RuntimeError("LOPO train file contains mixed domains")
    clean_rows = [{
        "sentence": x.sentence,
        "aspect_term": x.aspect_term,
        "label": x.label,
        "source_instance_id": x.source_instance_id,
        "cluster_id": x.cluster_id,
        "is_augmented": False,
        "perturbation_family": "Clean",
    } for x in examples]

    train_ids = {x.source_instance_id for x in examples}
    missing_selected = sorted(set(selected_ids) - train_ids)
    if missing_selected:
        raise RuntimeError(
            f"Frozen selected sources missing from train file: {missing_selected[:10]}"
        )

    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    core_conditions = core_manifest.get("conditions") or {}
    frozen_family_rows: dict[str, list[dict]] = {}
    frozen_family_hashes: dict[str, str] = {}
    for family in ("RevTgt", "RevNon", "AddDiff"):
        raw = family_paths.get(family)
        if raw is None:
            raise RuntimeError(f"Missing frozen family path for {family}")
        path = Path(raw)
        if not path.is_file():
            raise RuntimeError(f"Missing frozen single-family materialization for {family}: {path}")
        actual_sha = sha256_file(path)
        entry = core_conditions.get(f"Clean+{family}")
        if not isinstance(entry, dict):
            raise RuntimeError(f"Core transformation manifest lacks Clean+{family}")
        declared_sha = entry.get("sha256")
        if not declared_sha or declared_sha != actual_sha:
            raise RuntimeError(
                f"Frozen {family} file hash does not match core transformation manifest"
            )
        all_rows = read_jsonl(path)
        aug = [r for r in all_rows if bool(r.get("is_augmented"))]
        if len(aug) != expected_budget:
            raise RuntimeError(
                f"Frozen {family} augmented count mismatch: {len(aug)} != {expected_budget}"
            )
        aug_ids = [str(r.get("source_instance_id")) for r in aug]
        if Counter(aug_ids) != Counter(selected_ids):
            raise RuntimeError(
                f"Frozen {family} source composition differs from selected source pool"
            )
        wrong_family = [r for r in aug if r.get("perturbation_family") != family]
        if wrong_family:
            raise RuntimeError(f"Frozen {family} file contains mislabelled perturbation rows")
        frozen_family_rows[family] = aug
        frozen_family_hashes[family] = actual_sha

    if manifest_path.exists():
        manifest_path.unlink()

    conditions = materialize_lopo_from_frozen_rows(
        clean_rows, frozen_family_rows, selected_ids, seed
    )

    protected_sentences: set[str] = set()
    protected_hashes: dict[str, str] = {}
    for raw_path in protected_eval_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"Missing protected evaluation file: {path}")
        protected_hashes[str(path)] = sha256_file(path)
        for row in read_jsonl(path):
            if "sentence" not in row:
                raise RuntimeError(f"Protected evaluation row lacks sentence: {path}")
            protected_sentences.add(str(row["sentence"]))

    firewall_overlaps: dict[str, list[str]] = {}
    for condition, rows in conditions.items():
        overlap = sorted({str(row["sentence"]) for row in rows} & protected_sentences)
        if overlap:
            firewall_overlaps[condition] = overlap
    if firewall_overlaps:
        first_condition = sorted(firewall_overlaps)[0]
        raise RuntimeError(
            f"Protected-test sentence overlap in LOPO condition {first_condition}: "
            f"{firewall_overlaps[first_condition][:5]}"
        )

    validated: dict[str, dict] = {}
    selected_counter = Counter(selected_ids)
    for condition, rows in conditions.items():
        aug = [row for row in rows if bool(row.get("is_augmented"))]
        aug_ids = [str(row.get("source_instance_id")) for row in aug]
        family_counts = Counter(str(row.get("perturbation_family")) for row in aug)
        expected_family_counts = _family_quota_expected(expected_budget, condition)
        held_out = HELD_OUT[condition]
        if len(aug) != expected_budget:
            raise RuntimeError(f"LOPO augmentation budget mismatch for {condition}")
        if Counter(aug_ids) != selected_counter:
            raise RuntimeError(f"LOPO source composition mismatch for {condition}")
        if dict(family_counts) != expected_family_counts:
            raise RuntimeError(
                f"LOPO family quota mismatch for {condition}: "
                f"{dict(family_counts)} != {expected_family_counts}"
            )
        if held_out in family_counts:
            raise RuntimeError(f"LOPO held-out family leaked into {condition}: {held_out}")
        validated[condition] = {
            "n_rows": len(rows),
            "n_clean": len(rows) - len(aug),
            "n_augmented": len(aug),
            "unique_augmented_sources": len(set(aug_ids)),
            "family_counts": dict(family_counts),
            "held_out_family": held_out,
            "protected_sentence_overlap_count": 0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_conditions: dict[str, dict] = {}
    for condition in LOPO_CONDITIONS:
        path = out_dir / condition / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, conditions[condition])
        manifest_conditions[condition] = {
            **validated[condition],
            "path": str(path),
            "sha256": sha256_file(path),
        }

    manifest = {
        "protocol_stage": "leave_one_perturbation_out_minimal",
        "implementation": "frozen-single-family recombination LOPO v2",
        "source_mode": "reuse_frozen_single_family_materializations",
        "domain": domain,
        "seed": seed,
        "total_budget": expected_budget,
        "source_policy": "frozen_common_pool_exactly_once_per_condition",
        "odd_budget_policy": "rotate_single_extra_slot_symmetrically_across_families",
        "source_selection_path": str(selected_path),
        "source_selection_sha256": sha256_file(selected_path),
        "selected_ids_hash": hash_object(selected_ids),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "core_transformation_manifest_path": str(core_manifest_path),
        "core_transformation_manifest_sha256": sha256_file(core_manifest_path),
        "frozen_family_sha256": frozen_family_hashes,
        "protected_eval_sha256": protected_hashes,
        "conditions": manifest_conditions,
    }
    manifest["manifest_hash"] = hash_object(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--core-manifest", required=True)
    ap.add_argument(
        "--family-train",
        action="append",
        required=True,
        help="FAMILY=PATH for frozen Clean+RevTgt/RevNon/AddDiff train files; repeat three times.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--expected-budget", type=int, required=True)
    ap.add_argument(
        "--protected-eval",
        action="append",
        default=[],
        help="Protected evaluation JSONL; repeat for Clean/RevTgt/RevNon/AddDiff.",
    )
    args = ap.parse_args()

    family_paths: dict[str, str] = {}
    for item in args.family_train:
        if "=" not in item:
            raise RuntimeError(f"Malformed --family-train value: {item!r}")
        family, raw_path = item.split("=", 1)
        if family in family_paths:
            raise RuntimeError(f"Duplicate --family-train family: {family}")
        family_paths[family] = raw_path
    expected_families = {"RevTgt", "RevNon", "AddDiff"}
    if set(family_paths) != expected_families:
        raise RuntimeError(
            f"--family-train must provide exactly {sorted(expected_families)}; "
            f"found {sorted(family_paths)}"
        )

    manifest = materialize_domain_from_frozen(
        args.train,
        args.selected,
        args.core_manifest,
        family_paths,
        args.out_dir,
        seed=args.seed,
        expected_budget=args.expected_budget,
        protected_eval_paths=args.protected_eval,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
