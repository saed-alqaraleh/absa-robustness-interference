from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_campaign_lopo_produces_18_pinned_runs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    module = _load(repo / "scripts" / "11_build_campaign.py", "build_campaign_script")
    build = getattr(module, "build_campaign", None)
    assert callable(build), "scripts/11_build_campaign.py must expose build_campaign()"

    root = tmp_path / "project"
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        f"""project:\n  root: {root}\ndata:\n  domains: [restaurant, laptop]\nmodels:\n  minimal_decisive: FacebookAI/roberta-base\n  final: [google-bert/bert-base-uncased, FacebookAI/roberta-base, microsoft/deberta-v3-base]\n  pair_order: sentence_first\n  max_length: 128\ntraining:\n  learning_rate: 2.0e-5\n  train_batch_size: 16\n  epochs: 8\n  seeds_minimal: [13, 37, 73]\n  seeds_final: [13, 37, 73, 101, 137]\n""",
        encoding="utf-8",
    )
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"pins": {"FacebookAI/roberta-base": {"revision": "abc123"}}}), encoding="utf-8")
    out = tmp_path / "campaign"

    result = build(cfg, "lopo", out, pins)
    with Path(result["plan"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18
    assert {r["model"] for r in rows} == {"FacebookAI/roberta-base"}
    assert {int(r["seed"]) for r in rows} == {13, 37, 73}
    assert {r["model_revision"] for r in rows} == {"abc123"}

    commands = Path(result["commands"]).read_text(encoding="utf-8")
    assert commands.count("scripts/05_train_run.py") == 18
    assert "/usr/bin/python3 scripts/05_train_run.py" in commands
    assert "LOPO_MANIFEST.json" in commands


def test_lopo_preflight_module_exists_and_exposes_validate():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "16_validate_lopo_stage.py"
    assert path.is_file(), "LOPO live preflight script is required"
    module = _load(path, "lopo_preflight")
    assert callable(getattr(module, "validate", None))


def test_lopo_preflight_accepts_full_frozen_contract(tmp_path):
    from absa_xfer.campaign import LOPO_CONDITIONS, build_plan, make_train_command, write_plan_csv
    from absa_xfer.hashing import sha256_file
    from absa_xfer.io_utils import write_jsonl

    repo = Path(__file__).resolve().parents[1]
    validator = _load(repo / "scripts" / "16_validate_lopo_stage.py", "lopo_preflight_full")
    root = tmp_path / "root"
    (root / "code").mkdir(parents=True)

    budgets = {"restaurant": 1004, "laptop": 419}
    held = {
        "Clean+RevTgt+RevNon": "AddDiff",
        "Clean+RevTgt+AddDiff": "RevNon",
        "Clean+RevNon+AddDiff": "RevTgt",
    }
    odd_extra = {
        "Clean+RevTgt+RevNon": "RevTgt",
        "Clean+RevTgt+AddDiff": "AddDiff",
        "Clean+RevNon+AddDiff": "RevNon",
    }

    for domain, B in budgets.items():
        base = root / "data_processed" / domain
        base.mkdir(parents=True)
        write_jsonl(base / "dev.jsonl", [{"sentence": f"{domain} dev"}])
        write_jsonl(base / "test.jsonl", [{"sentence": f"{domain} protected clean"}])
        eval_dir = base / "evaluation"
        eval_dir.mkdir()
        for fam in ("RevTgt", "RevNon", "AddDiff"):
            write_jsonl(eval_dir / f"{fam}.jsonl", [{"sentence": f"{domain} protected {fam}"}])

        man_dir = root / "manifests" / domain
        man_dir.mkdir(parents=True)
        (man_dir / "split_manifest.json").write_text("{}\n", encoding="utf-8")

        selected_ids = [f"{domain}_{i}" for i in range(B)]
        sel = root / "audits" / "eligibility" / domain / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"
        sel.parent.mkdir(parents=True)
        sel.write_text(json.dumps({"budget": B, "selected_ids": selected_ids}), encoding="utf-8")

        lopo_root = root / "transformations" / "materialized" / domain
        lopo_root.mkdir(parents=True)
        core_manifest = {"conditions": {}}
        core_manifest_path = root / "manifests" / domain / "transformation_manifest.json"
        manifest = {
            "total_budget": B,
            "source_selection_sha256": sha256_file(sel),
            "source_mode": "reuse_frozen_single_family_materializations",
            "conditions": {},
            "frozen_family_sha256": {},
        }
        # Frozen single-family inputs and the core transformation manifest.
        for family in ("RevTgt", "RevNon", "AddDiff"):
            core_rows = [{
                "sentence": f"core generated {domain} {family} {sid}",
                "aspect_term": "x", "label": "positive", "source_instance_id": sid,
                "cluster_id": f"c_{sid}", "is_augmented": True,
                "perturbation_family": family, "generation": {},
            } for sid in selected_ids]
            cp = lopo_root / f"Clean+{family}" / "train.jsonl"
            cp.parent.mkdir(parents=True)
            write_jsonl(cp, core_rows)
            csha = sha256_file(cp)
            core_manifest["conditions"][f"Clean+{family}"] = {"sha256": csha}
            manifest["frozen_family_sha256"][family] = csha
        core_manifest_path.write_text(json.dumps(core_manifest), encoding="utf-8")
        manifest["core_transformation_manifest_sha256"] = sha256_file(core_manifest_path)

        for condition in LOPO_CONDITIONS:
            fa, fb = condition.removeprefix("Clean+").split("+")
            counts = {fa: B // 2, fb: B // 2}
            if B % 2:
                counts[odd_extra[condition]] += 1
            fams = []
            for fam, n in counts.items():
                fams.extend([fam] * n)
            assert len(fams) == B
            rows = [{
                "sentence": f"{domain} clean training sentence",
                "aspect_term": "x", "label": "positive", "source_instance_id": "clean",
                "cluster_id": "clean_cluster", "is_augmented": False,
                "perturbation_family": "Clean",
            }]
            for sid, fam in zip(selected_ids, fams):
                rows.append({
                    "sentence": f"generated {domain} {condition} {sid}",
                    "aspect_term": "x", "label": "positive", "source_instance_id": sid,
                    "cluster_id": f"c_{sid}", "is_augmented": True,
                    "perturbation_family": fam, "generation": {},
                })
            p = lopo_root / condition / "train.jsonl"
            p.parent.mkdir(parents=True)
            write_jsonl(p, rows)
            manifest["conditions"][condition] = {
                "sha256": sha256_file(p),
                "n_augmented": B,
                "held_out_family": held[condition],
            }
        (lopo_root / "LOPO_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = build_plan(
        ["restaurant", "laptop"],
        ["FacebookAI/roberta-base"],
        LOPO_CONDITIONS,
        [13, 37, 73],
        {"FacebookAI/roberta-base": {"revision": "abc123"}},
    )
    campaign_dir = root / "campaigns" / "lopo_minimal"
    campaign_dir.mkdir(parents=True)
    write_plan_csv(rows, campaign_dir / "lopo_plan.csv")
    hp = {"max_length": 128, "learning_rate": 2e-5, "train_batch_size": 16, "epochs": 8, "pair_order": "sentence_first"}
    commands = [make_train_command(r, root, hp, "lopo") for r in rows]
    (campaign_dir / "lopo_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n",
        encoding="utf-8",
    )

    audit = validator.validate(root, campaign_dir)
    assert audit["status"] == "READY_NOT_LAUNCHED"
    assert audit["n_plan_rows"] == 18
    assert audit["n_commands"] == 18
    assert audit["problems"] == []
