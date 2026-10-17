"""Offline tests for the Stage-1 kit: synthetic frozen root, tiny local RoBERTa, CPU only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "tests"))

from fixtures import make_root, make_tiny_model  # noqa: E402
from stage1 import losses as L  # noqa: E402
from stage1.common import CONDITIONS, POLAR_FLIP, read_jsonl, run_id, stage1_budget  # noqa: E402


def py(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(KIT / "stage1" / script), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s1")
    root = make_root(tmp)
    r = py("build_data.py", "--root", root, "--skip-robust-dev")
    assert r.returncode == 0, r.stdout + r.stderr
    return root, make_tiny_model(tmp)


def test_budget_rule():
    assert stage1_budget(1004, 875) == 1002 and stage1_budget(419, 397) == 414
    assert stage1_budget(1004, 300) == 600 and stage1_budget(419, 100) == 198


def test_build_data_constructions(built):
    root, _ = built
    audit = json.loads((root / "stage1_v1" / "data" / "PRE_RUN_AUDIT.json").read_text())
    assert audit["all_checks_pass"] and (root / "stage1_v1" / "data" / "_DATA_READY.json").exists()
    for d in ("restaurant", "laptop"):
        m = audit["domains"][d]
        assert m["pool_size"] == 32 and m["B1"] == 60          # 8 sources have an intensified RN
        assert m["pool_exclusions"] == {"no_reversal_only_RN": 8}
        c = m["conditions"]
        assert c["mixed"]["family_counts"] == {"RT": 20, "RN": 20, "AD": 20}
        assert c["triplet"]["n_sources"] == 20 and c["triplet"]["family_counts"] == {"RT": 20, "RN": 20, "AD": 20}
        assert c["bat"]["n_sources"] == 30 and c["bat"]["family_counts"] == {"TI": 30, "RT": 30}
        assert c["fei"]["n_sources"] == 20 and c["fei"]["family_counts"] == {"TI": 20, "RT": 20, "AD": 20}
        assert set(c["triplet"]["sources"]) <= set(c["bat"]["sources"])   # one common ordering for paired designs
        rows = read_jsonl(root / "stage1_v1" / "data" / d / "bat" / "train.jsonl")
        for r in rows:
            if r["stage1_family"] == "TI":
                assert r["label"] == r["orig_label"] and r["operator"].startswith("intensify:")
                assert any(f"{a} " in r["sentence"] for a in ("very", "really", "extremely"))
            if r["stage1_family"] == "RT":
                assert r["label"] == POLAR_FLIP[r["orig_label"]]


def test_deterministic_choice_matches_frozen_generator():
    from stage1.build_data import deterministic_choice
    released = Path(__file__).resolve().parents[3] / "study1_encoder_transfer" / "code" / "src"
    if not released.exists():
        pytest.skip("frozen generator source not available offline")
    sys.path.insert(0, str(released))
    from absa_xfer.transformations import deterministic_choice as frozen
    for k in range(50):
        key = f"20260913|src{k}|TgtInt|adv"
        assert deterministic_choice(["very", "really", "extremely"], key) == frozen(["very", "really", "extremely"], key)


def test_losses_numerics():
    import torch
    fam = torch.tensor([L.CLEAN, L.RT, L.RN, L.AD, L.CLEAN])
    group = torch.tensor([0, 0, 0, 0, -1])
    anchor = torch.tensor([1, 0, 0, 0, 0])
    logits = torch.tensor([[2.0, 0.0, -1.0], [-1.0, 0.0, 2.0], [2.0, 0.0, -1.0], [2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    assert float(L.invariance_loss(logits, fam, group, anchor)) == pytest.approx(0.0, abs=1e-6)   # RN, AD identical to anchor
    logits2 = logits.clone(); logits2[2] = torch.tensor([-1.0, 0.0, 2.0])
    assert float(L.invariance_loss(logits2, fam, group, anchor)) > 0.05
    labels = torch.tensor([0, 2, 0, 0, 1]); orig = torch.tensor([0, 0, 0, 0, 1])
    # RT item: z_{y'}=2, z_y=-1 -> gap 3 -> hinge(1-3)=0 ; with margin 4 -> 1
    assert float(L.sensitivity_loss(logits, fam, labels, orig, 1.0)) == pytest.approx(0.0)
    assert float(L.sensitivity_loss(logits, fam, labels, orig, 4.0)) == pytest.approx(1.0)
    # BAT: anchor e1, TI = e1 (d=0), RT = -e1 (d=2) -> [0 - 2 + 1]_+ = 0; swap -> 3
    reps = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    fb = torch.tensor([L.CLEAN, L.RT, L.TI]); gb = torch.tensor([0, 0, 0]); ab = torch.tensor([1, 0, 0])
    assert float(L.bat_triplet_loss(reps, fb, gb, ab, 1.0)) == pytest.approx(0.0)
    assert float(L.bat_triplet_loss(reps[[0, 2, 1]], fb, gb, ab, 1.0)) == pytest.approx(3.0)
    # Fei: sims TI=1, RT=-1, AD=0, mu=0.1 -> intra (-1-1)/0.1=-20 ; inter (0-1)/0.1=-10 -> -30
    reps_f = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    ff = torch.tensor([L.CLEAN, L.TI, L.RT, L.AD]); gf = torch.tensor([0, 0, 0, 0]); af = torch.tensor([1, 0, 0, 0])
    assert float(L.fei_opinion_loss(reps_f, ff, gf, af, 0.1)) == pytest.approx(-30.0, abs=1e-4)
    import math
    want = (math.log(math.exp(10) + math.exp(-10)) - 10) + (math.log(math.exp(10) + math.exp(0)) - 10)
    assert float(L.fei_opinion_loss(reps_f, ff, gf, af, 0.1, infonce=True)) == pytest.approx(want, abs=1e-4)


def test_groupdro_and_pcgrad():
    import torch
    dro = L.GroupDROWeights(4, eta=1.0)
    loss = dro.loss({0: (torch.tensor(0.1), 5), 1: (torch.tensor(2.0), 5)})
    assert dro.q[1] > dro.q[0] and float(loss) > 0.1
    g1, g2 = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    comb = L.pcgrad_combine([g1, g2], torch.Generator().manual_seed(0))
    assert torch.allclose(comb, torch.tensor([0.5, 1.5]))     # each projected off the other, then summed
    assert torch.allclose(L.pcgrad_combine([g1, torch.tensor([1.0, 1.0])], torch.Generator()), torch.tensor([2.0, 1.0]))


def test_units_cover_every_row_once(built):
    root, _ = built
    from stage1.trainer import make_units
    for data, grouped in (("triplet", True), ("mixed", True), ("mixed", False), ("clean_matched", False)):
        rows = read_jsonl(root / "stage1_v1" / "data" / "restaurant" / data / "train.jsonl")
        units = make_units(rows, grouped)
        flat = [i for u in units for i in u]
        assert sorted(flat) == list(range(len(rows)))
        for u in units:
            if len(u) > 1:
                assert not rows[u[0]]["is_augmented"] and all(rows[i]["group"] == rows[u[0]]["source_instance_id"] for i in u[1:])


@pytest.mark.parametrize("condition,cfg", [
    ("Mixed", {}), ("SCCT", {"lambda_i": 0.5, "lambda_s": 0.5, "margin": 1.0}), ("BAT", {"lambda_bat": 0.5, "margin": 1.0}),
    ("Fei", {"mu": 0.1, "lambda_f": 0.5}), ("Consistency", {"lambda_c": 0.5}), ("GroupDRO", {"eta": 0.1}), ("PCGrad", {}),
    ("CleanMatched", {}), ("Paired", {})])
def test_trainer_runs_every_method(built, tmp_path, condition, cfg):
    root, tiny = built
    out = tmp_path / "run"
    args = ["--root", root, "--condition", condition, "--domain", "laptop", "--seed", "13", "--config", json.dumps(cfg),
            "--phase", "confirm", "--out-dir", out, "--max-epochs", "2", "--model", tiny, "--revision", "main"]
    r = py("trainer.py", *args)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    res = json.loads((out / "run_result.json").read_text())
    assert set(res["eval_summary"]) == {"Clean", "RevTgt", "RevNon", "AddDiff"}
    assert (out / "predictions_RevNon.csv").exists() and len(res["history"]) >= 1
    r2 = py("trainer.py", *args)
    assert r2.returncode == 0 and "SKIP" in r2.stdout
    bad = list(args); bad[bad.index("13")] = "37"
    assert py("trainer.py", *bad).returncode == 1          # same out dir, different run -> refused


def test_tuning_phase_never_reads_arts(built, tmp_path):
    root, tiny = built
    out = tmp_path / "tune"
    r = py("trainer.py", "--root", root, "--condition", "SCCT", "--domain", "restaurant", "--seed", "101",
           "--config", json.dumps({"lambda_i": 1.0, "lambda_s": 1.0, "margin": 1.0}), "--phase", "tuning",
           "--out-dir", out, "--max-epochs", "1", "--model", tiny, "--revision", "main")
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((out / "run_result.json").read_text())
    assert set(res["eval_summary"]) == {"CleanDev", "DevRT", "DevRN", "DevAD"}


def test_plan_counts(built):
    root, _ = built
    assert py("run_plan.py", "plan", "--root", root, "--phase", "tuning").returncode == 1     # preflight not passed yet
    for phase, n in (("regression", 6), ("tuning", 62)):
        assert py("run_plan.py", "plan", "--root", root, "--phase", phase, "--skip-preflight-check").returncode == 0
        assert len(json.loads((root / "stage1_v1" / "plans" / f"{phase}_plan.json").read_text())["runs"]) == n
    assert py("run_plan.py", "plan", "--root", root, "--phase", "confirm").returncode == 1   # no frozen selection yet


def _fake_tuning(root: Path, best: dict) -> None:
    """Write tuning outputs where `best` configs win on S_dev and controls are clean-strong."""
    from stage1.common import CE_CONTROL, DOMAINS, TUNING_SEED, write_csv
    def write(cond, cfg, w, j, clean):
        for d in DOMAINS:
            rd = root / "stage1_v1" / "runs" / "tuning" / run_id(cond, d, TUNING_SEED, cfg)
            rd.mkdir(parents=True, exist_ok=True)
            ev = {"CleanDev": {"accuracy": clean}, "DevRT": {"accuracy": w + 0.05}, "DevRN": {"accuracy": w}, "DevAD": {"accuracy": w + 0.02}}
            (rd / "run_result.json").write_text(json.dumps({"eval_summary": ev}))
            n = 10
            k = round(j * n)
            write_csv(rd / "predictions_DevRT.csv", [{"source_instance_id": f"s{i}", "correct": 1} for i in range(n)])
            write_csv(rd / "predictions_DevRN.csv", [{"source_instance_id": f"s{i}", "correct": int(i < k)} for i in range(n)])
            (rd / "DONE.json").write_text("{}")
    for ctrl in set(CE_CONTROL.values()):
        write(ctrl, {}, 0.5, 0.5, 0.80)
    for cond in CE_CONTROL:
        for cfg in CONDITIONS[cond]["grid"]:
            win = cfg == best.get(cond)
            write(cond, cfg, 0.7 if win else 0.6, 0.7 if win else 0.6, 0.80)
    # a candidate that would win but breaks the clean guard
    bad = CONDITIONS["SCCT"]["grid"][-1]
    if bad != best["SCCT"]:
        write("SCCT", bad, 0.9, 0.9, 0.70)


def test_selection_rule(tmp_path):
    root = tmp_path / "root"
    best = {"SCCT": CONDITIONS["SCCT"]["grid"][4], "BAT": CONDITIONS["BAT"]["grid"][0], "Fei": CONDITIONS["Fei"]["grid"][2],
            "Consistency": CONDITIONS["Consistency"]["grid"][1], "GroupDRO": CONDITIONS["GroupDRO"]["grid"][2]}
    _fake_tuning(root, best)
    r = py("select_hparams.py", "--root", root)
    assert r.returncode == 0, r.stdout + r.stderr
    sel = json.loads((root / "stage1_v1" / "selection" / "SELECTED_CONFIGS.json").read_text())["selected"]
    for cond, cfg in best.items():
        assert sel[cond]["config"] == cfg, cond
    assert py("run_plan.py", "plan", "--root", root, "--phase", "confirm").returncode == 0
    plan = json.loads((root / "stage1_v1" / "plans" / "confirm_plan.json").read_text())["runs"]
    assert len(plan) == 54
    assert all(r["config"] == best["SCCT"] for r in plan if r["condition"] == "SCCT")


def _fake_confirm(root: Path, quality: dict) -> None:
    """quality[cond] = (p_rt, p_rn, p_ad, p_clean) correct-probability per family."""
    import random
    from stage1.common import CONFIRMATORY_SEEDS, DOMAINS, write_csv
    sel = {c: {"config": CONDITIONS[c]["grid"][0]} for c in CONDITIONS if CONDITIONS[c]["grid"] != [{}]}
    (root / "stage1_v1" / "selection").mkdir(parents=True, exist_ok=True)
    (root / "stage1_v1" / "selection" / "SELECTED_CONFIGS.json").write_text(json.dumps({"selected": sel}))
    from stage1.common import sha256_file
    (root / "stage1_v1" / "selection" / "_SELECTION_FROZEN.json").write_text(json.dumps(
        {"selected_sha256": sha256_file(root / "stage1_v1" / "selection" / "SELECTED_CONFIGS.json")}))
    for d in DOMAINS:
        for c in CONDITIONS:
            cfg = sel.get(c, {}).get("config", {})
            for s in CONFIRMATORY_SEEDS:
                twin = "SCCT" if (c == "Paired" and quality.get("Paired") == quality.get("SCCT")) else c
                rng = random.Random(f"{d}|{twin}|{s}")   # deterministic; identical twins share predictions
                rd = root / "stage1_v1" / "runs" / "confirm" / run_id(c, d, s, cfg)
                rd.mkdir(parents=True, exist_ok=True)
                q = quality.get(c, (0.6, 0.6, 0.6, 0.8))
                for name, p in zip(("RevTgt", "RevNon", "AddDiff", "Clean"), q):
                    n = 120 if name != "RevNon" else 60
                    write_csv(rd / f"predictions_{name}.csv",
                              [{"source_instance_id": f"src{i}", "cluster_id": f"cl{i // 2}", "correct": int(rng.random() < p)}
                               for i in range(n)])
                (rd / "DONE.json").write_text("{}")


@pytest.mark.parametrize("quality,expected", [
    ({"SCCT": (0.85, 0.85, 0.85, 0.8), "Paired": (0.62, 0.62, 0.62, 0.8)}, "SCCT_PASS"),
    ({"SCCT": (0.85, 0.85, 0.85, 0.8), "Paired": (0.85, 0.85, 0.85, 0.8)}, "PAIRED_DATA_CANDIDATE"),
    ({"SCCT": (0.61, 0.61, 0.61, 0.8), "Paired": (0.6, 0.6, 0.6, 0.8)}, "STOP_METHOD_PATH"),
    ({"SCCT": (0.85, 0.85, 0.85, 0.7), "Paired": (0.6, 0.6, 0.6, 0.8)}, "STOP_METHOD_PATH"),   # clean guardrail fails
])
def test_pass_rule_decisions(tmp_path, quality, expected):
    root = tmp_path / "root"
    _fake_confirm(root, quality)
    r = py("analyze.py", "--root", root, "--replicates", "300", "--expected-joint", "restaurant=60,laptop=60")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((root / "stage1_v1" / "analysis" / "STAGE1_DECISION.json").read_text())
    assert rep["decision"].startswith(expected), rep["decision"]
    if expected == "STOP_METHOD_PATH" and quality["SCCT"][3] == 0.7:
        assert rep["scct"]["criteria"]["D_clean_guardrail"] is False


def test_regression_check_reads_reference(tmp_path):
    from stage1.common import CONFIRMATORY_SEEDS, DOMAINS, KIT as K, read_csv
    root = tmp_path / "root"
    ref = {(r["domain"], r["evaluation_condition"], int(r["seed"])): float(r["value"])
           for r in read_csv(K / "reference" / "roberta_original_metrics.csv")
           if r["metric"] == "accuracy" and r["training_condition"] == "Clean+Mixed"}
    for d in DOMAINS:
        for s in CONFIRMATORY_SEEDS:
            rd = root / "stage1_v1" / "runs" / "regression" / run_id("Mixed", d, s, {})
            rd.mkdir(parents=True)
            ev = {f: {"accuracy": ref[(d, f, s)] + 0.003} for f in ("Clean", "RevTgt", "RevNon", "AddDiff")}
            (rd / "run_result.json").write_text(json.dumps({"eval_summary": ev}))
            (rd / "DONE.json").write_text("{}")
    r = py("regression_check.py", "--root", root)
    assert r.returncode == 0, r.stdout + r.stderr


def test_packing_keeps_full_batches_and_every_row():
    import random
    from stage1.trainer import make_batches
    units = [[i] for i in range(1000)] + [[1000 + 3 * k + j for j in range(3)] for k in range(100)] + \
            [[2000 + 4 * k + j for j in range(4)] for k in range(80)]
    batches = make_batches(units, 16, random.Random(1))
    n = sum(len(u) for u in units)
    flat = [i for b in batches for u in b for i in u]
    assert len(batches) == -(-n // 16) and sorted(flat) == sorted(i for u in units for i in u)
    assert all(sum(len(u) for u in b) <= 16 for b in batches)


def test_selection_refuses_undefined_scores(tmp_path):
    root = tmp_path / "root"
    best = {c: CONDITIONS[c]["grid"][0] for c in ("SCCT", "BAT", "Fei", "Consistency", "GroupDRO")}
    _fake_tuning(root, best)
    from stage1.common import TUNING_SEED, write_csv
    rd = root / "stage1_v1" / "runs" / "tuning" / run_id("SCCT", "laptop", TUNING_SEED, CONDITIONS["SCCT"]["grid"][0])
    write_csv(rd / "predictions_DevRN.csv", [{"source_instance_id": "other", "correct": 1}])   # no RT&RN overlap -> J undefined
    r = py("select_hparams.py", "--root", root)
    assert r.returncode == 1 and "Refusing to freeze" in r.stdout
    assert not (root / "stage1_v1" / "selection" / "_SELECTION_FROZEN.json").exists()


def test_reserved_tuning_pool(tmp_path):
    root = make_root(tmp_path, n_sentences=150)
    r = py("build_data.py", "--root", root, "--skip-robust-dev", "--tuning-set", "dev+reserved")
    assert r.returncode == 0, r.stdout + r.stderr
    audit = json.loads((root / "stage1_v1" / "data" / "PRE_RUN_AUDIT.json").read_text())
    for d in ("restaurant", "laptop"):
        m = audit["domains"][d]
        reserve = set(m["tuning_reserve"])
        assert len(reserve) == 36 and m["eligible_pool_size"] == 240 and m["pool_size"] == 204
        for name, c in m["conditions"].items():
            assert not reserve & set(c["sources"]), name               # never trained on
        rows = read_jsonl(root / "stage1_v1" / "data" / d / "robust_dev" / "RN.jsonl")
        assert {r["source_instance_id"] for r in rows} == reserve and m["robust_dev"]["adequate"]


def test_dev_tuning_set_too_small_blocks(tmp_path):
    root = make_root(tmp_path, n_sentences=20)
    r = py("build_data.py", "--root", root, "--skip-robust-dev", "--tuning-set", "dev+reserved")
    assert r.returncode == 1 and "tuning set too small" in r.stdout
    assert not (root / "stage1_v1" / "data" / "_DATA_READY.json").exists()


def test_code_change_invalidates_completed_run(built, tmp_path):
    root, tiny = built
    out = tmp_path / "run"
    args = ["--root", root, "--condition", "Mixed", "--domain", "laptop", "--seed", "13", "--config", "{}",
            "--phase", "confirm", "--out-dir", out, "--max-epochs", "1", "--model", tiny, "--revision", "main"]
    assert py("trainer.py", *args).returncode == 0
    done = json.loads((out / "DONE.json").read_text())
    done["signature"] = "stale-code"
    (out / "DONE.json").write_text(json.dumps(done))
    r = py("trainer.py", *args)
    assert r.returncode == 1 and "different completed run" in r.stdout
