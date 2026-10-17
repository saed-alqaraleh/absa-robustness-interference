from __future__ import annotations

import importlib.util
from pathlib import Path

from absa_xfer.campaign import CORE_CONDITIONS, build_plan, make_train_command, write_plan_csv


def load_validator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "14_validate_campaign_files.py"
    spec = importlib.util.spec_from_file_location("campaign_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def materialize_required_inputs(root: Path) -> None:
    for domain in ("restaurant", "laptop"):
        base = root / "data_processed" / domain
        base.mkdir(parents=True, exist_ok=True)
        for name in ("dev.jsonl", "test.jsonl"):
            (base / name).write_text("{}\n", encoding="utf-8")
        eval_dir = base / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for family in ("RevTgt", "RevNon", "AddDiff"):
            (eval_dir / f"{family}.jsonl").write_text("{}\n", encoding="utf-8")
        for condition in CORE_CONDITIONS:
            p = root / "transformations" / "materialized" / domain / condition / "train.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}\n", encoding="utf-8")
        manifest_dir = root / "manifests" / domain
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "split_manifest.json").write_text("{}\n", encoding="utf-8")
        (manifest_dir / "transformation_manifest.json").write_text("{}\n", encoding="utf-8")
        sel = root / "audits" / "eligibility" / domain / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"
        sel.parent.mkdir(parents=True, exist_ok=True)
        sel.write_text("{}\n", encoding="utf-8")
    (root / "code").mkdir(parents=True, exist_ok=True)


def build_campaign(root: Path) -> Path:
    rows = build_plan(
        ["restaurant", "laptop"],
        ["FacebookAI/roberta-base"],
        CORE_CONDITIONS,
        [13, 37, 73],
        {"FacebookAI/roberta-base": {"revision": "abc123"}},
    )
    campaign_dir = root / "campaigns" / "minimal_decisive"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_plan_csv(rows, campaign_dir / "minimal_plan.csv")
    hp = {
        "max_length": 128,
        "learning_rate": 2e-5,
        "train_batch_size": 16,
        "epochs": 8,
        "pair_order": "sentence_first",
    }
    commands = [make_train_command(row, root, hp, "minimal") for row in rows]
    (campaign_dir / "minimal_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    return campaign_dir


def test_campaign_file_preflight_accepts_correct_clean_and_arts_paths(tmp_path):
    materialize_required_inputs(tmp_path)
    campaign_dir = build_campaign(tmp_path)
    validator = load_validator()
    audit = validator.validate(tmp_path, campaign_dir)
    assert audit["status"] == "READY_TO_LAUNCH"
    assert audit["n_commands"] == 36
    assert audit["problems"] == []


def test_campaign_file_preflight_rejects_old_nonexistent_clean_path(tmp_path):
    materialize_required_inputs(tmp_path)
    campaign_dir = build_campaign(tmp_path)
    commands = campaign_dir / "minimal_commands.sh"
    text = commands.read_text(encoding="utf-8")
    text = text.replace(
        f"{tmp_path}/data_processed/restaurant/test.jsonl",
        f"{tmp_path}/data_processed/restaurant/evaluation/Clean.jsonl",
    )
    commands.write_text(text, encoding="utf-8")
    validator = load_validator()
    try:
        validator.validate(tmp_path, campaign_dir)
    except RuntimeError as exc:
        assert "BLOCKED" in str(exc)
        assert "Clean path mismatch" in str(exc)
    else:
        raise AssertionError("old evaluation/Clean.jsonl bug was not rejected")
