"""CPU tests of the Study IV kit on synthetic data and a tiny Qwen3 model (no GPU, no real data).

  python -m pytest -q tests/        (about 5 minutes on 2 CPU cores)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

from lora import analyze_lora as A  # noqa: E402
from lora.common import EXPECTED_J_SOURCES, jobs, select_jobs  # noqa: E402
from lora.data import load_eval, load_train  # noqa: E402
from lora.scoring import label_token_ids  # noqa: E402
from lora.train_lora import j_si, lr_lambda_factory  # noqa: E402

PY = sys.executable


@pytest.fixture(scope="session")
def syn(tmp_path_factory):
    base = tmp_path_factory.mktemp("syn")
    subprocess.run([PY, str(KIT / "tests" / "make_synthetic.py"), str(base)], check=True, capture_output=True)
    return {"root": base / "root", "model": base / "tiny_qwen3", "base": base}


def train(syn, *extra, out=None, check=True):
    cmd = [PY, "-u", str(KIT / "lora" / "train_lora.py"), "--root", str(syn["root"]), "--model", str(syn["model"]),
           "--allow-unpinned"] + (["--out-dir", str(out)] if out else []) + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)
    return r


# ----------------------------------------------------------------------------- units
def test_label_ids_drop_shared_first_tokens():
    class Tok:
        table = {"positive": [5], "Positive": [6], " positive": [7], " Positive": [9],
                 "negative": [1], "Negative": [2], " negative": [3], " Negative": [9],     # 9 shared
                 "neutral": [4, 1], "Neutral": [8], " neutral": [10], " Neutral": [11]}

        def encode(self, s, add_special_tokens=False):
            return self.table[s]
    ids = label_token_ids(Tok())
    assert ids == {"negative": [1, 2, 3], "neutral": [4, 8, 10, 11], "positive": [5, 6, 7]}


def test_j_si_joins_on_source():
    items = [{"set": "RevTgt", "source_instance_id": "a"}, {"set": "RevTgt", "source_instance_id": "b"},
             {"set": "RevTgt", "source_instance_id": "c"}, {"set": "RevNon", "source_instance_id": "b"},
             {"set": "RevNon", "source_instance_id": "a"}, {"set": "Clean", "source_instance_id": "a"}]
    assert j_si(items, [1, 1, 1, 0, 1, 0]) == (0.5, 2)


def test_holm_matches_hand_computation():
    adj, levels = A.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert levels == pytest.approx([1 - 0.05 / 3, 1 - 0.05 / 1, 1 - 0.05 / 2, 1 - 0.05 / 4])


def test_lr_schedule_and_sharding():
    f = lr_lambda_factory(100, 10)
    assert f(0) == 0 and f(10) == 1 and f(55) == pytest.approx(0.5) and f(100) == 0
    all_jobs = jobs()
    assert len(all_jobs) == 26
    shards = [select_jobs(all_jobs, f"{i}/3") for i in range(3)]
    assert sorted(map(str, sum(shards, []))) == sorted(map(str, all_jobs))


def test_frozen_size_checks(syn):
    assert len(load_train(syn["root"], "laptop", "Clean+Mixed")) == 2581
    with pytest.raises(SystemExit):
        load_train(syn["root"], "laptop", "Clean+Mixed", expected_rows=2580)
    ev = load_eval(syn["root"], "restaurant")
    assert len(ev) == 1120 + 846 + 444 + 1120


def _fake_runs(n_clusters=40, seeds=(13, 37, 73)):
    """Minimal in-memory runs: per seed, Clean-Matched and a condition with identical predictions."""
    import random
    rng = random.Random(1)
    base = []
    for k in range(n_clusters):
        for f in ("RevTgt", "RevNon", "AddDiff"):
            base.append({"set": f, "source_instance_id": f"s{k}", "cluster_id": f"c{k}", "gold": "positive"})
    runs = {}
    for s in seeds:
        corr = [str(rng.randint(0, 1)) for _ in base]
        for c in ("Clean-Matched", "Clean+RevTgt"):
            runs[("laptop", c, s)] = {"pred": [dict(r, correct=x) for r, x in zip(base, corr)]}
    return runs


def test_bootstrap_is_paired():
    import numpy as np
    runs = _fake_runs()
    dd = A.DomainData("laptop", runs, check_counts=False)
    W = A.resample_weights(dd, 500, np.random.default_rng(0))
    for s in (13, 37, 73):
        a, b = dd.stats("Clean+RevTgt", s, W), dd.stats("Clean-Matched", s, W)
        for e in ("RevTgt", "RevNon", "AddDiff", "J", "R_mean", "W"):
            assert np.all(a[e] - b[e] == 0)                  # identical predictions -> zero in every resample
    assert A.p_two_sided(np.zeros(500)) == 1.0


def test_bootstrap_detects_a_shift():
    import numpy as np
    runs = _fake_runs()
    for s in (13, 37, 73):                                   # condition loses 25% of correct RevNon items
        for k, r in enumerate(runs[("laptop", "Clean+RevTgt", s)]["pred"]):
            if r["set"] == "RevNon" and k % 4 == 0:
                runs[("laptop", "Clean+RevTgt", s)]["pred"][k] = dict(r, correct="0")
    dd = A.DomainData("laptop", runs, check_counts=False)
    W = A.resample_weights(dd, 2000, np.random.default_rng(0))
    boot = np.mean([dd.stats("Clean+RevTgt", s, W)["RevNon"] - dd.stats("Clean-Matched", s, W)["RevNon"]
                    for s in (13, 37, 73)], axis=0)
    assert boot.max() <= 0 and A.p_two_sided(boot) < 0.01


def test_item_order_mismatch_is_refused():
    runs = _fake_runs()
    runs[("laptop", "Clean+RevTgt", 37)]["pred"] = list(reversed(runs[("laptop", "Clean+RevTgt", 37)]["pred"]))
    with pytest.raises(SystemExit):
        A.DomainData("laptop", runs, check_counts=False)


# ----------------------------------------------------------------------------- integration
def test_signature_refusal_and_skip(syn, tmp_path):
    out = tmp_path / "zs"
    train(syn, "--domain", "laptop", "--condition", "ZS", "--seed", "0", out=out)
    r = train(syn, "--domain", "laptop", "--condition", "ZS", "--seed", "0", out=out)
    assert "SKIP" in r.stdout
    r = train(syn, "--domain", "laptop", "--condition", "ZS", "--seed", "0", "--eval-batch", "7", out=out)
    assert "SKIP" in r.stdout                                  # batch size is not part of the design
    r = train(syn, "--domain", "laptop", "--condition", "ZS", "--seed", "0", "--max-epochs", "1", out=out, check=False)
    assert r.returncode == 2 and "refusing to overwrite" in r.stdout
    before = (out / "predictions.csv").read_bytes()
    assert (out / "predictions.csv").read_bytes() == before


def test_resume_reproduces_uninterrupted_run(syn, tmp_path):
    args = ["--domain", "laptop", "--condition", "Clean+RevNon", "--seed", "37", "--max-train-items", "800",
            "--max-epochs", "3", "--eval-batch", "64"]
    ref = tmp_path / "ref"
    train(syn, *args, out=ref)
    res = tmp_path / "res"
    cmd = [PY, "-u", str(KIT / "lora" / "train_lora.py"), "--root", str(syn["root"]), "--model", str(syn["model"]),
           "--allow-unpinned", "--out-dir", str(res)] + args
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while not (res / "_resume" / "state.pt").exists() and p.poll() is None and time.time() - t0 < 300:
        time.sleep(0.02)
    p.kill()
    p.wait()
    assert not (res / "DONE.json").exists(), "run finished before it could be interrupted"
    r = train(syn, *args, out=res)
    assert "RESUME from epoch 1" in r.stdout
    assert (res / "predictions.csv").read_text() == (ref / "predictions.csv").read_text()
    a, b = (json.loads((d / "run_result.json").read_text()) for d in (ref, res))
    assert [h["dev_accuracy"] for h in a["history"]] == [h["dev_accuracy"] for h in b["history"]]
    assert not (res / "_resume").exists()


def test_end_to_end_plan_and_analysis(syn):
    root, model = syn["root"], syn["model"]
    rp = [PY, str(KIT / "lora" / "run_plan.py")]
    common = ["--root", str(root), "--model", str(model)]
    r = subprocess.run(rp + ["run"] + common, capture_output=True, text=True)
    assert r.returncode != 0 and "preflight" in (r.stdout + r.stderr)          # run refuses without preflight
    subprocess.run(rp + ["preflight"] + common + ["--revision", "synthetic-0", "--skip-gpu-checks"], check=True,
                   capture_output=True)
    targs = ["--train-args", "--max-train-items 48 --max-epochs 2"]
    for shard in ("0/2", "1/2"):
        r = subprocess.run(rp + ["run"] + common + ["--shard", shard] + targs, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = subprocess.run(rp + ["run"] + common + ["--shard", "0/1"] + targs, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.count("complete (DONE.json present)") == 26
    r = subprocess.run(rp + ["status"] + common, capture_output=True, text=True, check=True)
    assert "26 of 26 jobs complete" in r.stdout
    logs = list((root / "lora_qwen3_v1" / "logs").glob("*.log"))
    assert len(logs) == 26
    an = [PY, str(KIT / "lora" / "analyze_lora.py"), "--root", str(root), "--resamples", "2000"]
    subprocess.run(an, check=True, capture_output=True)
    res = json.loads((root / "lora_qwen3_v1" / "analysis" / "LORA_RESULTS.json").read_text())
    assert res["final"] and not res["missing_jobs"]
    assert {d: res["items"][d]["J_sources"] for d in res["items"]} == EXPECTED_J_SOURCES
    assert res["bootstrap"]["items_without_cluster_id"] == {"restaurant": 3, "laptop": 3}
    assert len(res["per_run"]) == 26 and len(res["L1"]["contrasts"]) == 4 and res["L1"]["seed_level_signs"]["total"] == 12
    assert res["L1"]["reading"] in ("reproduced", "partly reproduced", "not reproduced")
    for x in res["L1"]["contrasts"]:
        lo, hi = x["ci_holm"]
        u_lo, u_hi = x["ci95_unadjusted"]
        assert lo <= u_lo + 1e-12 and hi >= u_hi - 1e-12                         # adjusted interval is wider
    # --max-train-items 48 keeps only the first 48 rows, which are the shared clean base in every condition;
    # with the same seed the separately launched runs must therefore agree exactly (cross-process determinism)
    assert all(v["T"] == 0 for v in res["T"].values())
    # recompute one per-run endpoint by hand from predictions.csv
    import csv
    pr = next(r for r in res["per_run"] if r["domain"] == "laptop" and r["condition"] == "Clean+RevTgt" and r["seed"] == 37)
    rows = list(csv.DictReader(open(root / "lora_qwen3_v1" / "runs" / "laptop__Clean+RevTgt__seed37" / "predictions.csv")))
    rt = [int(r["correct"]) for r in rows if r["set"] == "RevTgt"]
    assert pr["RevTgt"] == pytest.approx(sum(rt) / len(rt))
    # determinism of the analysis
    first = (root / "lora_qwen3_v1" / "analysis" / "contrasts.csv").read_text()
    subprocess.run(an, check=True, capture_output=True)
    assert (root / "lora_qwen3_v1" / "analysis" / "contrasts.csv").read_text() == first
    shutil.copy(root / "lora_qwen3_v1" / "analysis" / "LORA_RESULTS.md", syn["base"] / "LORA_RESULTS_synthetic.md")
