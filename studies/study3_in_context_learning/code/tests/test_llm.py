"""Offline tests for the LLM probe kit (CPU, tiny random Llama, synthetic frozen root)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "tests"))

import fixtures  # noqa: E402
from llm import common  # noqa: E402
from llm.run_llm import (  # noqa: E402
    build_demos, label_token_ids, load_eval, load_model, render_template, score_job, split_prompt,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("llm")
    root = fixtures.make_root(tmp, n_sentences=30)
    model = fixtures.make_tiny_causal_lm(tmp, root)
    return root, model


# ----------------------------------------------------------------------------- demonstrations
def test_demos_structure_and_determinism(env):
    root, _ = env
    for d in common.DOMAINS:
        a, b = build_demos(root, d), build_demos(root, d)
        assert a == b
        assert len(a) == 5 * len(common.DEMO_SEEDS)
        test_ids = {r["source_instance_id"] for r in common.read_jsonl(root / "data_processed" / d / "test.jsonl")}
        rn = {r["source_instance_id"]: r for r in common.read_jsonl(
            root / "transformations" / "materialized" / d / "Clean+RevNon" / "train.jsonl") if r.get("is_augmented")}
        for key, demos in a.items():
            cond = key.split("__")[0]
            assert len(demos) == common.K
            srcs = [x["source_instance_id"] for x in demos]
            assert not set(srcs) & test_ids
            fams = [x["family"] for x in demos]
            labels = [x["label"] for x in demos]
            if cond in ("Clean", "RT", "RN"):
                assert set(fams) == {cond} and labels.count("positive") == 3 and labels.count("negative") == 3
                assert len(set(srcs)) == 6
            elif cond == "Mixed":
                assert sorted(fams) == ["AD", "AD", "RN", "RN", "RT", "RT"] and len(set(srcs)) == 6
            else:
                assert [fams[:3], fams[3:]] == [["RT", "RN", "AD"]] * 2
                assert len(set(srcs[:3])) == 1 and len(set(srcs[3:])) == 1 and srcs[0] != srcs[3]
            for x in demos:
                if x["family"] == "RN":
                    ops = rn[x["source_instance_id"]]["generation"]["metadata"]["operations"]
                    assert not any(o["operation"].startswith("intensify") for o in ops)
        assert a["Clean__seed13"] != a["Clean__seed37"]


def test_demo_labels_follow_family_rules(env):
    root, _ = env
    d = "restaurant"
    clean = {r["source_instance_id"]: r["label"] for r in common.read_jsonl(root / "data_processed" / d / "train.jsonl")}
    for key, demos in build_demos(root, d).items():
        for x in demos:
            orig = clean[x["source_instance_id"]]
            if x["family"] == "RT":
                assert x["label"] != orig
            else:
                assert x["label"] == orig


# ----------------------------------------------------------------------------- jobs and sharding
def test_jobs_and_sharding_partition():
    js = common.jobs(["m1", "m2"])
    assert len(js) == 2 * 2 * (1 + 5 * 3)
    assert len({(j["model"], common.job_id(j)) for j in js}) == len(js)
    parts = [common.select_jobs(js, f"{i}/2", f"{w}/3") for i in range(2) for w in range(3)]
    flat = [(j["model"], common.job_id(j)) for p in parts for j in p]
    assert sorted(flat) == sorted((j["model"], common.job_id(j)) for j in js)
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    with pytest.raises(SystemExit):
        common.select_jobs(js, "2/2")


def test_user_message_and_split(env):
    root, model = env
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    demos = build_demos(root, "laptop")["Mixed__seed37"]
    items = load_eval(root, "laptop")
    pre, post = split_prompt(tok, "laptop", demos)
    for it in items[:5]:
        assert pre + common.item_text(it) + post == render_template(tok, common.user_message("laptop", demos, it))
    zs = common.user_message("laptop", [], items[0])
    assert "Examples" not in zs and zs.endswith("Sentiment:")


def test_label_tokens_drop_collisions(env):
    _, model = env
    from transformers import AutoTokenizer
    ids = label_token_ids(AutoTokenizer.from_pretrained(model))
    flat = [i for v in ids.values() for i in v]
    assert len(flat) == len(set(flat)) and all(ids.values())


# ----------------------------------------------------------------------------- scoring equivalence
@pytest.mark.parametrize("cond", ["ZS", "Triplet__seed73"])
def test_prefix_cache_matches_full_prompt(env, cond):
    import torch
    root, model_dir = env
    tok, model = load_model(str(model_dir), torch.float32, torch.device("cpu"))
    lab = label_token_ids(tok)
    d = "restaurant"
    demos = [] if cond == "ZS" else build_demos(root, d)[cond]
    items = load_eval(root, d)[:23]
    full, nfb_full, _ = score_job(model, tok, d, demos, items, lab, 4, False, torch.device("cpu"))
    assert nfb_full == len(items)
    for bs in (1, 5, 32):
        cached, nfb, used = score_job(model, tok, d, demos, items, lab, bs, True, torch.device("cpu"))
        assert used == bs and nfb < len(items)
        assert torch.allclose(torch.tensor(cached), torch.tensor(full), atol=1e-4), bs
    # reference: one unpadded forward per item, no cache, no batching
    with torch.inference_mode():
        for it, sc in zip(items[:4], full):
            ids = tok(render_template(tok, common.user_message(d, demos, it)), add_special_tokens=False,
                      return_tensors="pt")["input_ids"]
            lp = torch.log_softmax(model(input_ids=ids).logits[0, -1], -1)
            ref = [torch.logsumexp(lp[lab[k]], -1).item() for k in common.LABELS]
            assert torch.allclose(torch.tensor(sc), torch.tensor(ref), atol=1e-4)


def test_deepcopy_cache_path_matches(env, monkeypatch):
    import torch
    from llm import run_llm
    root, model_dir = env
    tok, model = load_model(str(model_dir), torch.float32, torch.device("cpu"))
    lab = label_token_ids(tok)
    demos = build_demos(root, "laptop")["RN__seed37"]
    items = load_eval(root, "laptop")[:11]
    a, _, _ = score_job(model, tok, "laptop", demos, items, lab, 4, True, torch.device("cpu"))
    monkeypatch.setattr(run_llm, "_FORCE_DEEPCOPY", True)
    b, _, _ = score_job(model, tok, "laptop", demos, items, lab, 4, True, torch.device("cpu"))
    assert torch.allclose(torch.tensor(a), torch.tensor(b), atol=1e-5)


def test_oom_retry_halves_batch(env, monkeypatch):
    import torch
    from llm import run_llm
    root, model_dir = env
    tok, model = load_model(str(model_dir), torch.float32, torch.device("cpu"))
    lab = label_token_ids(tok)
    real = run_llm._score
    calls = []

    def flaky(*a, **k):
        calls.append(a[7])
        if a[7] > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return real(*a, **k)

    monkeypatch.setattr(run_llm, "_score", flaky)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    items = load_eval(root, "laptop")[:9]
    _, _, used = run_llm.score_job(model, tok, "laptop", [], items, lab, 16, True, torch.device("cpu"))
    assert calls == [16, 8, 4] and used == 4


# ----------------------------------------------------------------------------- end to end
def _run(args, check=True):
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")                    # offline tests always run on the CPU
    p = subprocess.run([sys.executable, str(KIT / "llm" / "run_llm.py"), *args], capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        raise AssertionError(p.stdout + p.stderr)
    return p


def test_end_to_end_shards_restart_and_analysis(env, tmp_path):
    root, model_dir = env
    m = str(model_dir)
    base = ["--root", str(root), "--models", m, "--batch-size", "8", "--out-subdir", "probe_test"]
    p0 = _run(base + ["--shard", "0/2", "--subshard", "0/2"])
    _run(base + ["--shard", "0/2", "--subshard", "1/2"])
    _run(base + ["--shard", "1/2"])
    odir = root / "probe_test" / common.slug(m)
    assert len(list(odir.glob("*.DONE.json"))) == 32 and len(list(odir.glob("*.csv"))) == 32
    assert "jobs" in p0.stdout
    before = {p.name: p.stat().st_mtime_ns for p in odir.iterdir()}
    again = _run(base)
    assert "already" in again.stdout or "complete" in again.stdout
    assert {p.name: p.stat().st_mtime_ns for p in odir.iterdir()} == before
    done = json.loads((odir / "restaurant__RT__seed13.DONE.json").read_text())
    n_items = sum(done["n"].values())
    assert n_items == len(load_eval(root, "restaurant"))
    with (odir / "restaurant__RT__seed13.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == n_items and {r["pred"] for r in rows} <= set(common.LABELS)
    # a run with a different configuration must refuse to reuse or overwrite the outputs
    bad = _run(base + ["--max-items", "3"], check=False)
    assert bad.returncode != 0 and "different configuration" in (bad.stdout + bad.stderr)
    # analysis on the synthetic outputs
    a = subprocess.run([sys.executable, str(KIT / "llm" / "analyze_llm.py"), "--root", str(root), "--out-subdir",
                        "probe_test", "--replicates", "200", "--expected-joint", "none", "--min-clusters", "1"],
                       capture_output=True, text=True)
    assert a.returncode == 0, a.stdout + a.stderr
    res = json.loads((root / "probe_test" / "analysis" / "LLM_PROBE_RESULTS.json").read_text())
    assert res["models_complete"] == [common.slug(m)]
    assert (root / "probe_test" / "analysis" / "LLM_PROBE_RESULTS.md").exists()


def test_analysis_recovers_known_effects(tmp_path):
    """Synthetic predictions with planted effects: RT demos cost 20 pp on RevNon, RN demos 10 pp on RevTgt."""
    import random
    from llm import analyze_llm
    root = tmp_path
    rng = random.Random(0)
    n_src = {"restaurant": 1500, "laptop": 1000}
    base_acc = {"Clean": 0.9, "RevTgt": 0.7, "RevNon": 0.8, "AddDiff": 0.85}
    shift = {("RT", "RevNon"): -0.20, ("RN", "RevTgt"): -0.10, ("Mixed", "RevNon"): 0.0}
    odir = root / common.OUT_SUBDIR / "fake__model"
    odir.mkdir(parents=True)
    for j in common.jobs(["fake/model"]):
        rows = []
        for s in common.EVAL_SETS:
            for i in range(n_src[j["domain"]]):
                p = base_acc[s] + shift.get((j["condition"], s), 0.0)
                c = int(rng.random() < p)
                rows.append({"set": s, "source_instance_id": f"{j['domain']}{i}", "cluster_id": f"c{i // 2}",
                             "gold": "positive", "pred": "positive" if c else "negative", "correct": c,
                             "s_negative": "-1", "s_neutral": "-3", "s_positive": "-0.5", "label_mass": "0.9"})
        with (odir / f"{common.job_id(j)}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (odir / f"{common.job_id(j)}.DONE.json").write_text(json.dumps({**j, "signature": {"max_items": None}}))
    res = analyze_llm.analyze(root, common.OUT_SUBDIR, replicates=500, seed=1, expected_joint=None, min_clusters=10)
    h2 = res["contrasts"]["fake__model"]
    assert abs(h2["H2_RN_acc_RTdemos_minus_Clean"]["estimate"] + 20) < 3
    assert abs(h2["H2_RT_acc_RNdemos_minus_Clean"]["estimate"] + 10) < 3
    assert h2["H2_RN_acc_RTdemos_minus_Clean"]["ci95"][1] < 0
    assert h2["H3_W_Mixed_minus_Clean"]["ci95"][0] < 0 < h2["H3_W_Mixed_minus_Clean"]["ci95"][1]
    zs = res["profile"]["fake__model"]["ZS"]
    assert abs(zs["W"]["estimate"] - 70) < 3 and abs(zs["Clean"]["estimate"] - 90) < 3


def test_preflight_mode(env):
    root, model_dir = env
    p = _run(["--root", str(root), "--models", str(model_dir), "--preflight", "--out-subdir", "probe_pf",
              "--min-label-mass", "0"])
    res = json.loads((root / "probe_pf" / "preflight" / f"{common.slug(str(model_dir))}.json").read_text())
    assert res["cached_vs_full_argmax_agreement"] == 1.0 and res["cached_vs_full_max_abs_diff"] < 1e-4
    assert res["share_items_with_clean_split"] == 1.0
    assert not list((root / "probe_pf").glob("*/*.DONE.json"))
    assert "preflight" in p.stdout


def test_rt_error_anatomy_runs(env):
    root, model_dir = env
    _run(["--root", str(root), "--models", str(model_dir), "--batch-size", "8", "--out-subdir", "probe_diag"])
    p = subprocess.run([sys.executable, str(KIT / "llm" / "diagnose_rt.py"), "--root", str(root), "--out-subdir", "probe_diag"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    res = json.loads((root / "probe_diag" / "analysis" / "RT_ERROR_ANATOMY.json").read_text())
    assert res["by_domain"]["restaurant"]["rt_items_with_original"] == 1.0
    for k, v in res["pooled"].items():
        if "RevTgt" in k and v["errors"]:
            assert abs(v["persist"] + v["neutral"] + v["other"] - 100) < 1e-6
