from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from absa_xfer.io_utils import write_jsonl


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "12_materialize_lopo.py"
    spec = importlib.util.spec_from_file_location("materialize_lopo_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_materialize_script_exposes_fail_closed_domain_function():
    module = _load_script()
    assert callable(getattr(module, "materialize_domain", None))


def test_materialize_domain_blocks_budget_mismatch_before_writing(tmp_path, monkeypatch):
    module = _load_script()
    assert callable(getattr(module, "materialize_domain", None))

    train = tmp_path / "train.jsonl"
    opinions = tmp_path / "opinions.jsonl"
    selected = tmp_path / "selection.json"
    out = tmp_path / "out"
    write_jsonl(train, [{
        "domain": "restaurant", "split": "train", "sentence_id": "s1",
        "sentence": "Great food.", "aspect_term": "food", "aspect_from": 6,
        "aspect_to": 10, "label": "positive", "source_instance_id": "a1",
        "cluster_id": "c1",
    }])
    write_jsonl(opinions, [{
        "source_instance_id": "a1", "cluster_id": "c1",
        "spans": [{"start": 0, "end": 5, "text": "Great", "provenance": "fixture"}],
        "matched_by": "fixture", "provenance": "fixture",
    }])
    selected.write_text(json.dumps({"budget": 1, "selected_ids": ["a1"]}), encoding="utf-8")

    try:
        module.materialize_domain(
            train, opinions, selected, out,
            seed=7, expected_budget=2, protected_eval_paths=[],
        )
    except RuntimeError as exc:
        assert "budget" in str(exc).lower()
    else:
        raise AssertionError("budget mismatch did not fail closed")
    assert not (out / "LOPO_MANIFEST.json").exists()


def test_materialize_domain_blocks_protected_sentence_overlap(tmp_path, monkeypatch):
    module = _load_script()
    assert callable(getattr(module, "materialize_domain", None))

    train = tmp_path / "train.jsonl"
    opinions = tmp_path / "opinions.jsonl"
    selected = tmp_path / "selection.json"
    protected = tmp_path / "protected.jsonl"
    out = tmp_path / "out"

    source = {
        "domain": "restaurant", "split": "train", "sentence_id": "s1",
        "sentence": "Great food.", "aspect_term": "food", "aspect_from": 6,
        "aspect_to": 10, "label": "positive", "source_instance_id": "a1",
        "cluster_id": "c1",
    }
    write_jsonl(train, [source])
    write_jsonl(opinions, [{
        "source_instance_id": "a1", "cluster_id": "c1",
        "spans": [{"start": 0, "end": 5, "text": "Great", "provenance": "fixture"}],
        "matched_by": "fixture", "provenance": "fixture",
    }])
    selected.write_text(json.dumps({"budget": 1, "selected_ids": ["a1"]}), encoding="utf-8")
    write_jsonl(protected, [{"sentence": "Great food."}])

    # Avoid needing the real transformation stack; the firewall must reject the
    # clean training sentence before any files are committed.
    fake_rows = [{
        "sentence": "Great food.", "aspect_term": "food", "label": "positive",
        "source_instance_id": "a1", "cluster_id": "c1", "is_augmented": False,
        "perturbation_family": "Clean",
    }, {
        "sentence": "Bad food.", "aspect_term": "food", "label": "negative",
        "source_instance_id": "a1", "cluster_id": "c1", "is_augmented": True,
        "perturbation_family": "RevTgt", "generation": {},
    }]
    monkeypatch.setattr(module, "materialize_lopo_conditions", lambda *args, **kwargs: {
        "Clean+RevTgt+RevNon": fake_rows,
        "Clean+RevTgt+AddDiff": fake_rows,
        "Clean+RevNon+AddDiff": fake_rows,
    })

    try:
        module.materialize_domain(
            train, opinions, selected, out,
            seed=7, expected_budget=1, protected_eval_paths=[protected],
        )
    except RuntimeError as exc:
        assert "protected" in str(exc).lower()
    else:
        raise AssertionError("protected-test overlap did not fail closed")
    assert not (out / "LOPO_MANIFEST.json").exists()


def test_selection_loader_accepts_final_generation_alias(tmp_path):
    module = _load_script()
    p = tmp_path / "selection.json"
    p.write_text(json.dumps({"budget": 2, "selected_source_ids": ["a", "b"]}), encoding="utf-8")
    ids, obj = module._load_selection(p)
    assert ids == ["a", "b"]
    assert obj["budget"] == 2


def test_materialize_domain_from_frozen_reuses_exact_family_rows(tmp_path):
    module = _load_script()
    fn = getattr(module, "materialize_domain_from_frozen", None)
    assert callable(fn), "production LOPO preparation must reuse frozen family rows"

    train = tmp_path / "train.jsonl"
    selected = tmp_path / "selection.json"
    protected = tmp_path / "protected.jsonl"
    core_manifest = tmp_path / "core_manifest.json"
    core_root = tmp_path / "core"
    out = tmp_path / "out"

    train_rows = []
    selected_ids = ["a", "b", "c"]
    for i, sid in enumerate(selected_ids):
        train_rows.append({
            "domain": "restaurant", "split": "train", "sentence_id": f"s{i}",
            "sentence": f"clean sentence {sid}", "aspect_term": "x", "aspect_from": 0,
            "aspect_to": 1, "label": "positive", "source_instance_id": sid,
            "cluster_id": f"c_{sid}",
        })
    write_jsonl(train, train_rows)
    selected.write_text(json.dumps({"budget": 3, "selected_ids": selected_ids}), encoding="utf-8")
    write_jsonl(protected, [{"sentence": "protected only"}])

    manifest_conditions = {}
    family_paths = {}
    from absa_xfer.hashing import sha256_file
    for fam in ("RevTgt", "RevNon", "AddDiff"):
        rows = [{
            "sentence": f"FROZEN::{fam}::{sid}", "aspect_term": "x", "label": "positive",
            "source_instance_id": sid, "cluster_id": f"c_{sid}",
            "is_augmented": True, "perturbation_family": fam,
            "generation": {"marker": f"{fam}:{sid}"},
        } for sid in selected_ids]
        p = core_root / f"Clean+{fam}" / "train.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(p, rows)
        family_paths[fam] = p
        manifest_conditions[f"Clean+{fam}"] = {"sha256": sha256_file(p)}
    core_manifest.write_text(json.dumps({"conditions": manifest_conditions}), encoding="utf-8")

    manifest = fn(
        train, selected, core_manifest, family_paths, out,
        seed=7, expected_budget=3, protected_eval_paths=[protected],
    )
    assert manifest["source_mode"] == "reuse_frozen_single_family_materializations"
    rows = list(__import__("absa_xfer.io_utils", fromlist=["read_jsonl"]).read_jsonl(out / "Clean+RevTgt+RevNon" / "train.jsonl"))
    aug = [r for r in rows if r["is_augmented"]]
    assert all(r["sentence"].startswith("FROZEN::") for r in aug)
    assert {r["source_instance_id"] for r in aug} == set(selected_ids)
