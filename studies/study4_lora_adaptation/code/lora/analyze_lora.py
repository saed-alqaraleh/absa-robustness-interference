#!/usr/bin/env python3
"""Study IV analysis: endpoints, paired cluster bootstrap, Holm correction and the pre-fixed readings of
L1 and L2 (protocol/LORA_PROTOCOL_v1.0.md).

  python lora/analyze_lora.py --root ROOT              # requires all 26 jobs
  python lora/analyze_lora.py --root ROOT --partial    # interim view of whatever is complete (not final)

Reads only <root>/lora_qwen3_v1/runs/*/predictions.csv and run_result.json; writes
<root>/lora_qwen3_v1/analysis/{LORA_RESULTS.json, LORA_RESULTS.md, per_run.csv, contrasts.csv}.

Bootstrap: ARTS sentence clusters are resampled within each domain (10,000 resamples, seed 20261007).
One resample matrix per domain is applied to every run of that domain, so all contrasts are paired.
Seeds are fixed: each statistic is the seed mean of seed-matched differences; domains are then averaged
with equal weight (each domain resampled independently). p-values are two-sided,
p = min(1, 2 * min(#(T* <= 0), #(T* >= 0)) / B), floored at 1/B. Holm-adjusted intervals use, for the
contrast ranked r (0-based) by p, the percentile interval at level 1 - 0.05 / (m - r).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora.common import (  # noqa: E402
    ARTS_FAMILIES, CONDITIONS, DOMAINS, EXPECTED_J_SOURCES, KIT_VERSION, SEEDS, ZS, atomic_write_json,
    atomic_write_text, jobs, log, out_root, protocol_sha256, run_name, sha256_file,
)

B_DEFAULT = 10_000
BOOT_SEED = 20261007
ALPHA = 0.05
CM = "Clean-Matched"
TRAIN_FAMILY = {"Clean+RevTgt": "RevTgt", "Clean+RevNon": "RevNon"}
PRIMARY = [("restaurant", "Clean+RevTgt", "RevNon"), ("restaurant", "Clean+RevNon", "RevTgt"),
           ("laptop", "Clean+RevTgt", "RevNon"), ("laptop", "Clean+RevNon", "RevTgt")]


# ----------------------------------------------------------------------------- loading
def read_predictions(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_runs(root: Path, partial: bool) -> dict:
    runs, missing = {}, []
    for j in jobs():
        d = out_root(root) / "runs" / run_name(j)
        if not (d / "DONE.json").exists():
            missing.append(run_name(j))
            continue
        done = json.loads((d / "DONE.json").read_text())
        if sha256_file(d / "predictions.csv") != done["predictions_sha256"]:
            raise SystemExit(f"{d}/predictions.csv does not match the digest in DONE.json")
        runs[(j["domain"], j["condition"], j["seed"])] = {
            "result": json.loads((d / "run_result.json").read_text()), "pred": read_predictions(d / "predictions.csv")}
    if missing and not partial:
        raise SystemExit(f"{len(missing)} jobs incomplete (first: {missing[0]}); finish them or use --partial")
    revs = {r["result"]["revision"] for r in runs.values()}
    if len(revs) > 1:
        raise SystemExit(f"runs use different model revisions: {revs}")
    return {"runs": runs, "missing": missing, "revision": revs.pop() if revs else None}


class DomainData:
    """Cluster-level sufficient statistics of one domain; identical item order is asserted for all runs."""

    def __init__(self, d: str, runs: dict, check_counts: bool):
        import numpy as np
        keys = [k for k in runs if k[0] == d]
        if not keys:
            raise SystemExit(f"no completed runs for {d}")
        ref = runs[keys[0]]["pred"]
        sig = [(r["set"], r["source_instance_id"], r["gold"]) for r in ref]
        for k in keys[1:]:
            if [(r["set"], r["source_instance_id"], r["gold"]) for r in runs[k]["pred"]] != sig:
                raise SystemExit(f"{d}: item order or gold labels differ between {keys[0]} and {k}; pairing invalid")
        self.d = d
        arts = [i for i, r in enumerate(ref) if r["set"] in ARTS_FAMILIES]
        self.fallback_clusters = sum(1 for i in arts if not ref[i]["cluster_id"])
        ckey = [ref[i]["cluster_id"] or f"__src__{ref[i]['source_instance_id']}" for i in arts]
        names = sorted(set(ckey))
        self.n_clusters = len(names)
        pos = {c: k for k, c in enumerate(names)}
        self.item_cluster = {i: pos[c] for i, c in zip(arts, ckey)}
        self.fam_items = {f: [i for i in arts if ref[i]["set"] == f] for f in ARTS_FAMILIES}
        self.n = {f: len(v) for f, v in self.fam_items.items()}
        # joint sources: source_instance_id present in both RevTgt and RevNon (must share a cluster)
        rt = {ref[i]["source_instance_id"]: i for i in self.fam_items["RevTgt"]}
        rn = {ref[i]["source_instance_id"]: i for i in self.fam_items["RevNon"]}
        self.pairs = [(rt[s], rn[s]) for s in sorted(set(rt) & set(rn))]
        for a, b in self.pairs:
            if self.item_cluster[a] != self.item_cluster[b]:
                raise SystemExit(f"{d}: RevTgt/RevNon items of one source fall in different clusters")
        if check_counts and len(self.pairs) != EXPECTED_J_SOURCES[d]:
            raise SystemExit(f"{d}: {len(self.pairs)} joint sources, expected {EXPECTED_J_SOURCES[d]}")
        # per-cluster denominators
        self.den = {f: np.bincount([self.item_cluster[i] for i in self.fam_items[f]], minlength=self.n_clusters)
                    for f in ARTS_FAMILIES}
        self.den["J"] = np.bincount([self.item_cluster[a] for a, _ in self.pairs], minlength=self.n_clusters)
        self.num = {}
        for k in keys:
            corr = [int(r["correct"]) for r in runs[k]["pred"]]
            v = {f: np.bincount([self.item_cluster[i] for i in self.fam_items[f]],
                                weights=[corr[i] for i in self.fam_items[f]], minlength=self.n_clusters)
                 for f in ARTS_FAMILIES}
            v["J"] = np.bincount([self.item_cluster[a] for a, _ in self.pairs],
                                 weights=[corr[a] * corr[b] for a, b in self.pairs], minlength=self.n_clusters)
            v["Clean"] = sum(corr[i] for i, r in enumerate(ref) if r["set"] == "Clean") / max(
                1, sum(1 for r in ref if r["set"] == "Clean"))
            self.num[k[1:]] = v                               # key (condition, seed)

    def stats(self, cond: str, seed: int, W=None) -> dict:
        """Endpoints of one run; W = resample weight matrix [B, clusters] (None = observed data)."""
        v = self.num[(cond, seed)]
        out = {}
        for f in ARTS_FAMILIES + ("J",):
            if W is None:
                out[f] = v[f].sum() / self.den[f].sum()
            else:
                out[f] = (W @ v[f]) / (W @ self.den[f])
        import numpy as np
        fam = np.stack([out[f] for f in ARTS_FAMILIES]) if W is not None else np.array([out[f] for f in ARTS_FAMILIES])
        out["R_mean"] = fam.mean(axis=0)
        out["W"] = fam.min(axis=0)
        return out


def resample_weights(dd: DomainData, B: int, rng):
    import numpy as np
    idx = rng.integers(0, dd.n_clusters, size=(B, dd.n_clusters))
    W = np.zeros((B, dd.n_clusters), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(B), dd.n_clusters), idx.ravel()), 1.0)
    return W


# ----------------------------------------------------------------------------- inference helpers
def p_two_sided(boot) -> float:
    import numpy as np
    B = len(boot)
    p = 2 * min(np.sum(boot <= 0), np.sum(boot >= 0)) / B
    return float(min(1.0, max(p, 1.0 / B)))


def interval(boot, level: float) -> list[float]:
    import numpy as np
    a = 1 - level
    return [float(np.quantile(boot, a / 2)), float(np.quantile(boot, 1 - a / 2))]


def holm(ps: list[float]) -> tuple[list[float], list[float]]:
    """Holm-adjusted p-values and, per contrast, the interval level 1 - alpha / (m - rank)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, levels, run = [0.0] * m, [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
        levels[i] = 1 - ALPHA / (m - r)
    return adj, levels


# ----------------------------------------------------------------------------- analysis
def analyse(root: Path, partial: bool, B: int, check_counts: bool) -> dict:
    import numpy as np
    loaded = load_runs(root, partial)
    runs = loaded["runs"]
    dom = {d: DomainData(d, runs, check_counts) for d in DOMAINS if any(k[0] == d for k in runs)}
    rng = np.random.default_rng(BOOT_SEED)
    Wd = {d: resample_weights(dom[d], B, rng) for d in DOMAINS if d in dom}   # fixed order: restaurant, laptop

    def seeds_for(d, cond):
        return [s for s in SEEDS if (d, cond, s) in runs]

    # per-run table
    per_run = []
    for (d, c, s), r in sorted(runs.items()):
        st = dom[d].stats(c, s)
        per_run.append({"domain": d, "condition": c, "seed": s, "Clean": dom[d].num[(c, s)]["Clean"],
                        **{f: float(st[f]) for f in ARTS_FAMILIES}, "R_mean": float(st["R_mean"]),
                        "W": float(st["W"]), "J_SI": float(st["J"]), "best_epoch": r["result"]["best_epoch"],
                        "best_dev_accuracy": r["result"]["best_dev_accuracy"],
                        "train_minutes": round(r["result"]["train_seconds"] / 60, 1)})

    def contrast(d, cond, endpoint):
        """Seed-mean of seed-matched (cond − Clean-Matched): (observed, bootstrap vector, per-seed values)."""
        ss = [s for s in seeds_for(d, cond) if (d, CM, s) in runs]
        if not ss:
            return None
        key = "J" if endpoint == "J_SI" else endpoint
        obs = [float(dom[d].stats(cond, s)[key] - dom[d].stats(CM, s)[key]) for s in ss]
        boot = np.mean([dom[d].stats(cond, s, Wd[d])[key] - dom[d].stats(CM, s, Wd[d])[key] for s in ss], axis=0)
        return float(np.mean(obs)), boot, dict(zip(ss, obs))

    # descriptive profiles (seed mean and SD)
    profiles = {}
    for d in dom:
        for c in (ZS,) + CONDITIONS:
            rows = [r for r in per_run if r["domain"] == d and r["condition"] == c]
            if rows:
                profiles[f"{d}/{c}"] = {e: {"mean": float(np.mean([r[e] for r in rows])),
                                            "sd": float(np.std([r[e] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                                            "n_seeds": len(rows)}
                                        for e in ("Clean",) + ARTS_FAMILIES + ("R_mean", "W", "J_SI")}

    # full transfer matrix T (descriptive, unadjusted)
    T = {}
    for d in dom:
        for c in CONDITIONS[1:]:
            for f in ARTS_FAMILIES:
                res = contrast(d, c, f)
                if res:
                    obs, boot, per_seed = res
                    T[f"{d}/{c}/{f}"] = {"T": obs, "ci95": interval(boot, 0.95), "p": p_two_sided(boot),
                                         "per_seed": per_seed}

    # L1: four directed reciprocal contrasts, Holm
    l1 = []
    for d, c, f in PRIMARY:
        res = contrast(d, c, f) if d in dom else None
        if res:
            obs, boot, per_seed = res
            l1.append({"domain": d, "train": TRAIN_FAMILY[c], "test": f, "T": obs, "p": p_two_sided(boot),
                       "per_seed": per_seed, "_boot": boot})
    if l1:
        adj, levels = holm([x["p"] for x in l1])
        for x, a, lv in zip(l1, adj, levels):
            x["p_holm"] = a
            x["holm_level"] = lv
            x["ci_holm"] = interval(x["_boot"], lv)
            x["ci95_unadjusted"] = interval(x["_boot"], 0.95)
            del x["_boot"]
    complete_l1 = len(l1) == 4 and all(len(x["per_seed"]) == len(SEEDS) for x in l1)
    if not complete_l1:
        verdict = "not evaluable (incomplete runs)"
    elif all(x["T"] < 0 for x in l1) and all(x["ci_holm"][1] < 0 for x in l1):
        verdict = "reproduced"
    elif all(x["T"] < 0 for x in l1):
        verdict = "partly reproduced"
    else:
        verdict = "not reproduced"
    seed_signs = {"negative": sum(v < 0 for x in l1 for v in x["per_seed"].values()),
                  "zero": sum(v == 0 for x in l1 for v in x["per_seed"].values()),
                  "positive": sum(v > 0 for x in l1 for v in x["per_seed"].values()),
                  "total": sum(len(x["per_seed"]) for x in l1)}

    # L2: Mixed − Clean-Matched, equal-domain mean
    l2 = {}
    for e in ("R_mean", "W", "J_SI"):
        per_dom = {d: contrast(d, "Clean+Mixed", e) for d in dom}
        if all(per_dom.get(d) for d in DOMAINS):
            obs = float(np.mean([per_dom[d][0] for d in DOMAINS]))
            boot = np.mean([per_dom[d][1] for d in DOMAINS], axis=0)
            l2[e] = {"diff_equal_domain": obs, "ci95": interval(boot, 0.95), "p": p_two_sided(boot),
                     "by_domain": {d: {"diff": per_dom[d][0], "ci95": interval(per_dom[d][1], 0.95),
                                       "p": p_two_sided(per_dom[d][1])} for d in DOMAINS}}
    l2_verdict = ("not evaluable (incomplete runs)" if len(l2) < 3 else
                  "supported" if all(v["diff_equal_domain"] > 0 and v["ci95"][0] > 0 for v in l2.values()) else
                  "partly supported" if any(v["diff_equal_domain"] > 0 and v["ci95"][0] > 0 for v in l2.values()) else
                  "not supported")

    # one-sided training: change in R_mean (descriptive)
    rmean_change = {}
    for c in ("Clean+RevTgt", "Clean+RevNon"):
        per_dom = {d: contrast(d, c, "R_mean") for d in dom}
        rmean_change[c] = {d: {"diff": v[0], "ci95": interval(v[1], 0.95)} for d, v in per_dom.items() if v}

    return {
        "kit_version": KIT_VERSION, "protocol_sha256": protocol_sha256(), "model_revision": loaded["revision"],
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "final": not loaded["missing"],
        "missing_jobs": loaded["missing"], "bootstrap": {"B": B, "seed": BOOT_SEED, "unit": "ARTS sentence cluster",
                                                         "clusters": {d: dom[d].n_clusters for d in dom},
                                                         "items_without_cluster_id": {d: dom[d].fallback_clusters for d in dom}},
        "items": {d: {**dom[d].n, "J_sources": len(dom[d].pairs)} for d in dom},
        "per_run": per_run, "profiles": profiles, "T": T,
        "L1": {"contrasts": l1, "reading": verdict, "seed_level_signs": seed_signs},
        "L2": {"contrasts": l2, "reading": l2_verdict}, "R_mean_change_one_sided": rmean_change,
    }


# ----------------------------------------------------------------------------- reporting
def pct(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci(c) -> str:
    return f"[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]"


def markdown(res: dict) -> str:
    L = [f"# Study IV results: LoRA fine-tuning of Qwen3-8B (kit {res['kit_version']})", ""]
    if not res["final"]:
        L += [f"**INTERIM: {len(res['missing_jobs'])} jobs incomplete. Not for reporting.**", ""]
    L += [f"Model revision `{res['model_revision']}`; protocol SHA-256 `{res['protocol_sha256'][:16]}…`; "
          f"generated {res['generated']}.",
          f"Bootstrap: {res['bootstrap']['B']:,} resamples of ARTS sentence clusters "
          f"(seed {res['bootstrap']['seed']}; clusters {res['bootstrap']['clusters']}; "
          f"items without cluster_id, resampled as their own source: {res['bootstrap']['items_without_cluster_id']}).",
          "Differences are in percentage points.", "",
          "## L1: reciprocal interference (primary, Holm over four contrasts)", "",
          "| Domain | Train | Test | T | Holm-adjusted CI | p (Holm) | per seed |", "|---|---|---|---|---|---|---|"]
    for x in res["L1"]["contrasts"]:
        seeds = ", ".join(f"{s}: {pct(v)}" for s, v in x["per_seed"].items())
        L.append(f"| {x['domain']} | {x['train']} | {x['test']} | {pct(x['T'])} | {ci(x['ci_holm'])} "
                 f"({100 * x['holm_level']:.2f}%) | {x['p_holm']:.4f} | {seeds} |")
    s = res["L1"]["seed_level_signs"]
    L += ["", f"Reading (fixed in advance): **{res['L1']['reading']}**. Seed-level signs: "
              f"{s['negative']} negative, {s['zero']} zero, {s['positive']} positive of {s['total']}.", "",
          "## L2: co-exposure (Clean+Mixed − Clean-Matched, equal-domain mean)", "",
          "| Endpoint | Difference | 95% CI | p | Restaurant | Laptop |", "|---|---|---|---|---|---|"]
    for e, v in res["L2"]["contrasts"].items():
        bd = v["by_domain"]
        L.append(f"| {e} | {pct(v['diff_equal_domain'])} | {ci(v['ci95'])} | {v['p']:.4f} | "
                 f"{pct(bd['restaurant']['diff'])} {ci(bd['restaurant']['ci95'])} | "
                 f"{pct(bd['laptop']['diff'])} {ci(bd['laptop']['ci95'])} |")
    L += ["", f"Reading: **{res['L2']['reading']}**.", "",
          "## Transfer matrix T (descriptive, unadjusted 95% CI)", "",
          "| Domain | Condition | Family | T | 95% CI | p |", "|---|---|---|---|---|---|"]
    for k, v in res["T"].items():
        d, c, f = k.split("/")
        L.append(f"| {d} | {c} | {f} | {pct(v['T'])} | {ci(v['ci95'])} | {v['p']:.4f} |")
    L += ["", "## Profiles (accuracy, mean ± SD over seeds; ZS = zero-shot reference, one deterministic run)", "",
          "| Domain/condition | Clean | RevTgt | RevNon | AddDiff | R_mean | W | J_SI |", "|---|---|---|---|---|---|---|---|"]
    for k, v in res["profiles"].items():
        L.append(f"| {k} | " + " | ".join(f"{100 * v[e]['mean']:.2f} ± {100 * v[e]['sd']:.2f}"
                                          for e in ("Clean",) + ARTS_FAMILIES + ("R_mean", "W", "J_SI")) + " |")
    return "\n".join(L) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    import io
    if not rows:
        return
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--partial", action="store_true", help="interim view; output marked INTERIM")
    ap.add_argument("--resamples", type=int, default=B_DEFAULT, help="keep 10000 for reporting")
    ap.add_argument("--no-count-check", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    root = args.root.expanduser().resolve()
    res = analyse(root, args.partial, args.resamples, not args.no_count_check)
    out = out_root(root) / "analysis"
    atomic_write_json(out / "LORA_RESULTS.json", res)
    atomic_write_text(out / "LORA_RESULTS.md", markdown(res))
    write_csv(out / "per_run.csv", res["per_run"])
    write_csv(out / "contrasts.csv", [{"contrast": k, **{kk: v[kk] for kk in ("T", "p")},
                                       "ci_low": v["ci95"][0], "ci_high": v["ci95"][1]} for k, v in res["T"].items()])
    log("analysis", f"L1: {res['L1']['reading']}; L2: {res['L2']['reading']}; written to {out}")
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
