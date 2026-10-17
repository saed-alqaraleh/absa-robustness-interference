from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "15_prepare_lopo_stage.py"
    assert path.is_file(), "LOPO preparation orchestrator is required"
    spec = importlib.util.spec_from_file_location("prepare_lopo_stage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_stage_requires_explicit_continue_decision():
    module = _load()
    gate = getattr(module, "validate_continue_decision", None)
    assert callable(gate)
    assert gate("CONTINUE") == "CONTINUE"
    for bad in ("STOP", "REASSESS", "", None):
        try:
            gate(bad)
        except RuntimeError as exc:
            assert "continue" in str(exc).lower()
        else:
            raise AssertionError(f"decision {bad!r} should have blocked LOPO preparation")


def test_prepare_stage_materialize_command_reuses_frozen_family_files(tmp_path):
    module = _load()
    builder = getattr(module, "build_materialize_command", None)
    assert callable(builder), "prepare stage must expose frozen-family materialization command builder"
    root = tmp_path / "root"
    cmd = builder(root, "restaurant", 1004, "/usr/bin/python3")
    joined = " ".join(cmd)
    assert "--core-manifest" in cmd
    assert "--opinions" not in cmd
    for family in ("RevTgt", "RevNon", "AddDiff"):
        expected = root / "transformations" / "materialized" / "restaurant" / f"Clean+{family}" / "train.jsonl"
        assert f"{family}={expected}" in cmd
