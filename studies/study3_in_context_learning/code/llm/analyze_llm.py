#!/usr/bin/env python3
"""Analyse the LLM probe: ARTS profiles, pre-specified contrasts H1-H3, cluster bootstrap.

Endpoints follow Study I / Stage 1 exactly: accuracy per set (Clean, RevTgt, RevNon, AddDiff),
W = min(A_RevTgt, A_RevNon, A_AddDiff), J = share of ARTS sources with an RevNon item whose RevTgt and
RevNon items are both correct. Per model and condition, endpoints are averaged over demonstration
draws (seeds), then over the two domains with equal weight. Intervals: percentile 95% from a
cluster bootstrap over ARTS sentence clusters within each domain; the same resample is applied to
every model and condition, so all contrasts are paired. "Model mean" averages the complete models.

If the Stage-1 confirmatory RoBERTa runs exist under <root>/stage1_v1/runs/confirm, CleanMatched,
Mixed and Paired are added as fine-tuned references on the same items and resamples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.common import (  # noqa: E402
    CONDITIONS, DEMO_SEEDS, DOMAINS, EVAL_SETS, OUT_SUBDIR, atomic_write_json, atomic_write_text, job_id, jobs, log,
)

REF_CONDS = ("CleanMatched", "Mixed", "Paired")
REF_PREFIX = "RoBERTa-ft"
METRICS = ("Clean", "RevTgt", "RevNon", "AddDiff", "W", "J")

# Pre-specified contrasts: name -> (family, [(weight, condition, metric)], predicted sign or 0 for two-sided)
CONTRASTS = {
    # H1: zero-shot LLMs are not ARTS-robust (W below clean accuracy); profile of the gaps
    "H1_W_minus_Clean_ZS": ("H1", [(1, "ZS", "W"), (-1, "ZS", "Clean")], -1),
    "H1_RevTgt_minus_RevNon_ZS": ("H1", [(1, "ZS", "RevTgt"), (-1, "ZS", "RevNon")], 0),
    "H1_RevTgt_minus_Clean_ZS": ("H1d", [(1, "ZS", "RevTgt"), (-1, "ZS", "Clean")], 0),
    "H1_RevNon_minus_Clean_ZS": ("H1d", [(1, "ZS", "RevNon"), (-1, "ZS", "Clean")], 0),
    "H1_AddDiff_minus_Clean_ZS": ("H1d", [(1, "ZS", "AddDiff"), (-1, "ZS", "Clean")], 0),
    # H2: reciprocal in-context interference relative to exposure-matched clean demonstrations
    "H2_RN_acc_RTdemos_minus_Clean": ("H2", [(1, "RT", "RevNon"), (-1, "Clean", "RevNon")], -1),
    "H2_RT_acc_RNdemos_minus_Clean": ("H2", [(1, "RN", "RevTgt"), (-1, "Clean", "RevTgt")], -1),
    "H2_RT_acc_RTdemos_minus_Clean": ("H2d", [(1, "RT", "RevTgt"), (-1, "Clean", "RevTgt")], 0),
    "H2_RN_acc_RNdemos_minus_Clean": ("H2d", [(1, "RN", "RevNon"), (-1, "Clean", "RevNon")], 0),
    "H2_W_RTdemos_minus_Clean": ("H2d", [(1, "RT", "W"), (-1, "Clean", "W")], 0),
    "H2_W_RNdemos_minus_Clean": ("H2d", [(1, "RN", "W"), (-1, "Clean", "W")], 0),
    # H3: co-exposure recovers balance
    "H3_W_Mixed_minus_Clean": ("H3", [(1, "Mixed", "W"), (-1, "Clean", "W")], 1),
    "H3_J_Mixed_minus_Clean": ("H3", [(1, "Mixed", "J"), (-1, "Clean", "J")], 1),
    "H3_W_Triplet_minus_Clean": ("H3", [(1, "Triplet", "W"), (-1, "Clean", "W")], 1),
    "H3_J_Triplet_minus_Clean": ("H3", [(1, "Triplet", "J"), (-1, "Clean", "J")], 1),
    "H3_W_Triplet_minus_Mixed": ("H3d", [(1, "Triplet", "W"), (-1, "Mixed", "W")], 0),
    "H3_J_Triplet_minus_Mixed": ("H3d", [(1, "Triplet", "J"), (-1, "Mixed", "J")], 0),
    # context: effect of clean demonstrations themselves
    "CTX_W_Clean_minus_ZS": ("CTX", [(1, "Clean", "W"), (-1, "ZS", "W")], 0),
    "CTX_J_Clean_minus_ZS": ("CTX", [(1, "Clean", "J"), (-1, "ZS", "J")], 0),
}
HOLM_FAMILIES = ("H1", "H2", "H3")


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def seeds_of(cond: str) -> tuple[int, ...]:
    return (0,) if cond == "ZS" else DEMO_SEEDS


class DomainData:
    """Per-cluster sufficient statistics so every bootstrap replicate is a matrix product."""

    def __init__(self, runs: dict[tuple, dict[str, list[dict]]]):
        clusters = sorted({r["cluster_id"] for rr in runs.values() for rows in rr.values() for r in rows})
        self.k = {c: i for i, c in enumerate(clusters)}
        self.K = len(clusters)
        self.stats = {}
        for key, rr in runs.items():
            st = {}
            for s in EVAL_SETS:
                c, n = np.zeros(self.K), np.zeros(self.K)
                for r in rr[s]:
                    i = self.k[r["cluster_id"]]
                    c[i] += int(r["correct"])
                    n[i] += 1
                st[s] = (c, n)
            rt = {r["source_instance_id"]: int(r["correct"]) for r in rr["RevTgt"]}
            jc, jn = np.zeros(self.K), np.zeros(self.K)
            for r in rr["RevNon"]:
                if r["source_instance_id"] in rt:
                    i = self.k[r["cluster_id"]]
                    jc[i] += int(r["correct"]) * rt[r["source_instance_id"]]
                    jn[i] += 1
            st["J"] = (jc, jn)
            self.stats[key] = st

    def endpoints(self, key, w: np.ndarray) -> dict[str, np.ndarray]:
        st = self.stats[key]
        acc = {s: 100 * (w @ st[s][0]) / np.maximum(w @ st[s][1], 1e-12) for s in EVAL_SETS}
        return {**acc, "W": np.minimum(np.minimum(acc["RevTgt"], acc["RevNon"]), acc["AddDiff"]),
                "J": 100 * (w @ st["J"][0]) / np.maximum(w @ st["J"][1], 1e-12)}


def split_by_set(rows: list[dict]) -> dict[str, list[dict]]:
    out = {s: [] for s in EVAL_SETS}
    for r in rows:
        out[r["set"]].append(r)
    return out


def summarize(v: np.ndarray) -> dict:
    rep = v[1:]
    return {"estimate": float(v[0]), "ci95": [float(np.percentile(rep, 2.5)), float(np.percentile(rep, 97.5))]}


def p_two_sided(v: np.ndarray) -> float:
    rep = v[1:]
    return float(min(1.0, 2 * min(np.mean(rep <= 0), np.mean(rep >= 0))))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[k]))
        out[k] = run
    return out


def analyze(root: Path, out_subdir: str = OUT_SUBDIR, replicates: int = 10000, seed: int = 20261007,
            expected_joint: dict | None = None, min_clusters: int = 50, include_reference: bool = True) -> dict:
    out_root = root / out_subdir
    job_ids = [job_id(j) for j in jobs(["x"])]
    models, incomplete, sigs = [], {}, {}
    for mdir in sorted(p for p in out_root.iterdir() if p.is_dir() and any(p.glob("*.DONE.json"))):
        done = [mdir / f"{j}.DONE.json" for j in job_ids]
        missing = [p.name for p in done if not p.exists() or not p.with_name(p.name.replace(".DONE.json", ".csv")).exists()]
        if missing:
            incomplete[mdir.name] = len(missing)
            continue
        s = {json.dumps(json.loads(p.read_text(encoding="utf-8")).get("signature"), sort_keys=True) for p in done}
        if len(s) != 1:
            raise SystemExit(f"{mdir.name}: jobs were produced with different configurations")
        sigs[mdir.name] = json.loads(s.pop())
        models.append(mdir.name)
    if not models:
        raise SystemExit(f"no complete model under {out_root} (incomplete: {incomplete})")
    full_run = all((sigs[m] or {}).get("max_items") is None for m in models)
    log("analyze", f"complete models: {models}; incomplete: {incomplete or 'none'}")

    data, diag = {}, defaultdict(dict)
    ref_present = []
    for d in DOMAINS:
        runs = {}
        for m in models:
            for c in CONDITIONS:
                for s in seeds_of(c):
                    j = {"domain": d, "condition": c, "seed": s}
                    rows = read_csv(out_root / m / f"{job_id(j)}.csv")
                    runs[(m, c, s)] = split_by_set(rows)
                    dj = json.loads((out_root / m / f"{job_id(j)}.DONE.json").read_text(encoding="utf-8"))
                    dd = diag[m].setdefault(c, defaultdict(list))
                    dd["label_mass"] += [float(r["label_mass"]) for r in rows if "label_mass" in r]
                    for st in EVAL_SETS:
                        dd[f"neutral_share_{st}"].append(np.mean([r["pred"] == "neutral" for r in runs[(m, c, s)][st]]))
                    dd["fallback_items"].append(dj.get("fallback_items", 0))
                    dd["seconds"].append(dj.get("seconds", 0))
        confirm = root / "stage1_v1" / "runs" / "confirm"
        if include_reference and confirm.exists():
            for c in REF_CONDS:
                got = {}
                for s in DEMO_SEEDS:
                    hits = [p for p in confirm.glob(f"{c}__{d}__*__seed{s}") if (p / "DONE.json").exists()]
                    if len(hits) == 1:
                        got[s] = {st: read_csv(hits[0] / f"predictions_{st}.csv") for st in EVAL_SETS}
                if len(got) == len(DEMO_SEEDS):
                    for s, rr in got.items():
                        runs[(REF_PREFIX, c, s)] = rr
                    ref_present.append((d, c))
        dd_ = DomainData(runs)
        for key, v in dd_.stats.items():
            n_joint = int(v["J"][1].sum())
            if full_run and expected_joint and expected_joint.get(d) is not None and n_joint != expected_joint[d]:
                raise SystemExit(f"{d} {key}: RevTgt/RevNon joint set has {n_joint} sources, expected {expected_joint[d]}")
        if "" in dd_.k or dd_.K < min_clusters:
            raise SystemExit(f"{d}: {dd_.K} clusters (or empty cluster ids); the cluster bootstrap would be invalid")
        data[d] = dd_
    refs = [c for c in REF_CONDS if all((d, c) in ref_present for d in DOMAINS)]

    rng = np.random.default_rng(seed)
    weights = {}
    for d in DOMAINS:
        K = data[d].K
        Wm = np.ones((replicates + 1, K))
        draws = rng.integers(0, K, size=(replicates, K))
        for r in range(replicates):
            Wm[r + 1] = np.bincount(draws[r], minlength=K)
        weights[d] = Wm

    def cell(owner: str, c: str, d: str) -> dict[str, np.ndarray]:
        eps = [data[d].endpoints((owner, c, s), weights[d]) for s in (DEMO_SEEDS if owner == REF_PREFIX else seeds_of(c))]
        return {k: np.mean([e[k] for e in eps], axis=0) for k in METRICS}

    def per_seed(owner: str, c: str, d: str, s: int) -> dict[str, float]:
        e = data[d].endpoints((owner, c, s), weights[d][:1])
        return {k: float(e[k][0]) for k in METRICS}

    E = {m: {c: {d: cell(m, c, d) for d in DOMAINS} for c in CONDITIONS} for m in models}
    pooled = {m: {c: {k: np.mean([E[m][c][d][k] for d in DOMAINS], axis=0) for k in METRICS} for c in CONDITIONS}
              for m in models}
    pooled["model_mean"] = {c: {k: np.mean([pooled[m][c][k] for m in models], axis=0) for k in METRICS}
                            for c in CONDITIONS}
    ref_pooled = {c: {k: np.mean([cell(REF_PREFIX, c, d)[k] for d in DOMAINS], axis=0) for k in METRICS} for c in refs}

    profile = {}
    for owner in list(models) + ["model_mean"]:
        profile[owner] = {}
        for c in CONDITIONS:
            profile[owner][c] = {k: summarize(pooled[owner][c][k]) for k in METRICS}
            if owner != "model_mean":
                for k in METRICS:
                    profile[owner][c][k]["by_domain"] = {d: float(E[owner][c][d][k][0]) for d in DOMAINS}
    reference = {f"{REF_PREFIX}:{c}": {k: summarize(ref_pooled[c][k]) for k in METRICS} for c in refs}

    contrasts = {}
    for owner in list(models) + ["model_mean"]:
        contrasts[owner] = {}
        for name, (fam, terms, sign) in CONTRASTS.items():
            v = sum(w * pooled[owner][c][k] for w, c, k in terms)
            res = {**summarize(v), "family": fam, "predicted_sign": sign, "p_two_sided": p_two_sided(v)}
            cells = []
            for m in (models if owner == "model_mean" else [owner]):
                for d in DOMAINS:
                    for s in ((0,) if all(c == "ZS" for _, c, _ in terms) else DEMO_SEEDS):
                        val = 0.0
                        for w, c, k in terms:
                            val += w * per_seed(m, c, d, s if c != "ZS" else 0)[k]
                        cells.append(val)
            res["cells"] = len(cells)
            res["cells_negative"] = int(sum(x < 0 for x in cells))
            res["cells_positive"] = int(sum(x > 0 for x in cells))
            contrasts[owner][name] = res
        for fam in HOLM_FAMILIES:
            ps = {n: r["p_two_sided"] for n, r in contrasts[owner].items() if r["family"] == fam}
            for n, p in holm(ps).items():
                contrasts[owner][n]["p_holm"] = p
    vs_ref = {}
    for c in refs:
        for owner in list(models) + ["model_mean"]:
            for k in ("W", "J", "Clean"):
                vs_ref[f"{owner}:ZS_minus_{REF_PREFIX}:{c}:{k}"] = summarize(pooled[owner]["ZS"][k] - ref_pooled[c][k])

    diagnostics = {}
    for m in models:
        diagnostics[m] = {}
        for c, dd in diag[m].items():
            lm = np.array(dd["label_mass"]) if dd["label_mass"] else np.array([math.nan])
            diagnostics[m][c] = {"label_mass_mean": float(np.mean(lm)), "label_mass_below_0.5": float(np.mean(lm < 0.5)),
                                 **{k: float(np.mean(v)) for k, v in dd.items() if k.startswith("neutral_share")},
                                 "fallback_items": int(sum(dd["fallback_items"])), "gpu_seconds": float(sum(dd["seconds"]))}
    return {"models_complete": models, "models_incomplete": incomplete, "full_run": full_run, "signatures": sigs,
            "replicates": replicates, "bootstrap_seed": seed, "clusters": {d: data[d].K for d in DOMAINS},
            "references": refs, "profile": profile, "reference_profile": reference, "contrasts": contrasts,
            "zero_shot_vs_finetuned": vs_ref, "diagnostics": diagnostics}


def short(m: str) -> str:
    return m.split("__")[-1]


def fmt(x: dict, ci: bool = True) -> str:
    s = f"{x['estimate']:.1f}"
    return s + (f" [{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]" if ci else "")


def to_markdown(res: dict) -> str:
    L = ["# LLM probe of reciprocal robustness interference on ARTS", ""]
    L.append(f"Complete models: {', '.join(res['models_complete'])}. Full run: {res['full_run']}. "
             f"Cluster bootstrap: {res['replicates']} resamples (clusters {res['clusters']}). Values in percentage points; "
             "equal-domain means; demonstration draws averaged.")
    if res["models_incomplete"]:
        L.append(f"\nIncomplete models (left out): {res['models_incomplete']}")
    L += ["", "## A. Zero-shot ARTS profile (with fine-tuned RoBERTa references)", "",
          "| System | Clean | RevTgt | RevNon | AddDiff | W | J |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in res["models_complete"] + ["model_mean"]:
        p = res["profile"][m]["ZS"]
        L.append(f"| {short(m)} (zero-shot) | " + " | ".join(fmt(p[k]) for k in METRICS) + " |")
    for r, p in res["reference_profile"].items():
        L.append(f"| {r} | " + " | ".join(fmt(p[k]) for k in METRICS) + " |")
    L += ["", "## B. W and J by demonstration condition", "",
          "| System | " + " | ".join(f"{c} W | {c} J" for c in CONDITIONS) + " |",
          "| --- |" + " --- | --- |" * len(CONDITIONS)]
    for m in res["models_complete"] + ["model_mean"]:
        L.append(f"| {short(m)} | " + " | ".join(f"{fmt(res['profile'][m][c]['W'], False)} | {fmt(res['profile'][m][c]['J'], False)}"
                                          for c in CONDITIONS) + " |")
    L += ["", "## C. Accuracy per ARTS set by condition (model mean)", "",
          "| Condition | " + " | ".join(METRICS) + " |", "| --- |" + " --- |" * len(METRICS)]
    for c in CONDITIONS:
        L.append(f"| {c} | " + " | ".join(fmt(res["profile"]["model_mean"][c][k]) for k in METRICS) + " |")
    L += ["", "## D. Pre-specified contrasts (model mean; Holm within H1, H2, H3)", "",
          "| Contrast | Estimate [95% CI] | p (Holm) | Cells − / + | " + " | ".join(short(m) for m in res["models_complete"]) + " |",
          "| --- | --- | --- | --- |" + " --- |" * len(res["models_complete"])]
    for name, r in res["contrasts"]["model_mean"].items():
        ph = f"{r['p_holm']:.4f}" if "p_holm" in r else f"({r['p_two_sided']:.4f})"
        per = " | ".join(fmt(res["contrasts"][m][name]) for m in res["models_complete"])
        L.append(f"| {name} | {fmt(r)} | {ph} | {r['cells_negative']} / {r['cells_positive']} of {r['cells']} | {per} |")
    if res["zero_shot_vs_finetuned"]:
        L += ["", "## E. Zero-shot LLM minus fine-tuned RoBERTa (same items, paired resamples)", "",
              "| Comparison | Estimate [95% CI] |", "| --- | --- |"]
        for k, v in res["zero_shot_vs_finetuned"].items():
            L.append(f"| {k} | {fmt(v)} |")
    L += ["", "## F. Diagnostics", "", "| Model | Condition | Label mass (mean) | Share < 0.5 | Neutral share (Clean/RT/RN/AD) | Fallback items |",
          "| --- | --- | --- | --- | --- | --- |"]
    for m, dd in res["diagnostics"].items():
        for c, v in dd.items():
            ns = "/".join(f"{v.get(f'neutral_share_{s}', float('nan')):.2f}" for s in EVAL_SETS)
            L.append(f"| {short(m)} | {c} | {v['label_mass_mean']:.3f} | {v['label_mass_below_0.5']:.3f} | {ns} | {v['fallback_items']} |")
    L += ["", "Cells −/+ count model × domain × demonstration-draw cells with a negative / positive contrast. "
          "p values are two-sided bootstrap p values; families marked 'd' in the JSON are descriptive (no Holm).", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-subdir", default=OUT_SUBDIR)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261007)
    ap.add_argument("--expected-joint", default="restaurant=444,laptop=135", help="'none' disables the check")
    ap.add_argument("--min-clusters", type=int, default=50)
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args(argv)
    root = args.root.expanduser().resolve()
    ej = None if args.expected_joint == "none" else {k: int(v) for k, v in (x.split("=") for x in args.expected_joint.split(","))}
    res = analyze(root, args.out_subdir, args.replicates, args.bootstrap_seed, ej, args.min_clusters, not args.no_reference)
    out = root / args.out_subdir / "analysis"
    atomic_write_json(out / "LLM_PROBE_RESULTS.json", res)
    md = to_markdown(res)
    atomic_write_text(out / "LLM_PROBE_RESULTS.md", md)
    print(md)
    log("analyze", f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
