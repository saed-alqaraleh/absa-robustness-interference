#!/usr/bin/env python3
"""Stage 1: compute the primary endpoints and apply the frozen pass rule (protocol v1.0, Decision 4).

Per confirmatory run (ARTS test sets):
  A_f  accuracy on RevTgt, RevNon, AddDiff;  W = min(A_RT, A_RN, A_AD)
  J    share of ARTS sources with an RevNon item whose RevTgt and RevNon items are both correct
  C    clean (SemEval test) accuracy

Per condition and domain the three seeds are averaged; domains get equal weight.
The strongest comparator for each endpoint is the member of {Mixed, BAT, Fei} with the
highest equal-domain mean. Uncertainty: 95% percentile intervals from 10,000 source-cluster
bootstrap replicates (ARTS sentence clusters resampled within each domain; the same
resample is applied to every run, so comparisons stay paired).

Rule for a candidate method M (SCCT, and Paired for the fallback clause):
  A  dW >= 2.0 pp (equal-domain mean) and bootstrap lower bound > 0
  B  dJ >= 2.0 pp and bootstrap lower bound > 0
  C  in each domain, dW >= -1.0 and dJ >= -1.0 pp
  D  clean accuracy of M minus Mixed >= -1.0 pp (equal-domain mean)
  E  at least 2 of 3 seeds with equal-domain dW > 0 and dJ > 0
Decision: if Paired passes and SCCT does not improve on Paired (dW > 0 and dJ > 0), the
paired-data construction is the candidate; otherwise SCCT passing means proceed; if
neither passes, stop the method path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.common import (  # noqa: E402
    CONDITIONS, CONFIRMATORY_SEEDS, DOMAINS, STAGE1_SUBDIR, atomic_write_json, atomic_write_text, check_protocol, log,
    read_csv, run_id,
)

CORE = ("Mixed", "BAT", "Fei")
FAMS = ("RevTgt", "RevNon", "AddDiff")


def load_run(run_dir: Path) -> dict[str, list[dict]]:
    return {s: read_csv(run_dir / f"predictions_{s}.csv") for s in ("Clean",) + FAMS}


class DomainData:
    """Per-cluster sufficient statistics so every bootstrap replicate is a matrix product."""

    def __init__(self, runs: dict[tuple[str, int], dict[str, list[dict]]]):
        clusters = sorted({r["cluster_id"] for rr in runs.values() for rows in rr.values() for r in rows})
        self.k = {c: i for i, c in enumerate(clusters)}
        self.K = len(clusters)
        self.stats = {}
        for key, rr in runs.items():
            st = {}
            for s, rows in rr.items():
                c, n = np.zeros(self.K), np.zeros(self.K)
                for r in rows:
                    i = self.k[r["cluster_id"]]
                    c[i] += int(r["correct"])
                    n[i] += 1
                st[s] = (c, n)
            rt = {r["source_instance_id"]: (int(r["correct"]), r["cluster_id"]) for r in rr["RevTgt"]}
            jc, jn = np.zeros(self.K), np.zeros(self.K)
            for r in rr["RevNon"]:
                sid = r["source_instance_id"]
                if sid in rt:
                    i = self.k[r["cluster_id"]]
                    jc[i] += int(r["correct"]) * rt[sid][0]
                    jn[i] += 1
            st["J"] = (jc, jn)
            self.stats[key] = st

    def endpoints(self, key, weights: np.ndarray) -> dict[str, np.ndarray]:
        st = self.stats[key]
        acc = {s: (weights @ st[s][0]) / np.maximum(weights @ st[s][1], 1e-12) for s in ("Clean",) + FAMS}
        return {"W": 100 * np.minimum(np.minimum(acc["RevTgt"], acc["RevNon"]), acc["AddDiff"]),
                "J": 100 * (weights @ st["J"][0]) / np.maximum(weights @ st["J"][1], 1e-12),
                "C": 100 * acc["Clean"], **{f: 100 * acc[f] for f in FAMS}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261007)
    ap.add_argument("--expected-joint", default="restaurant=444,laptop=135",
                    help="ARTS sources with both RevTgt and RevNon items, per domain (sanity check)")
    ap.add_argument("--min-clusters", type=int, default=50)
    args = ap.parse_args()
    check_protocol()
    root = args.root.expanduser().resolve()
    runs_dir = root / STAGE1_SUBDIR / "runs" / "confirm"
    out = root / STAGE1_SUBDIR / "analysis"
    from stage1.run_plan import load_frozen_selection
    sel = load_frozen_selection(root)
    conds = list(CONDITIONS)
    required = ("Mixed", "BAT", "Fei", "Paired", "SCCT")
    expected_joint = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in args.expected_joint.split(","))
    missing, missing_optional = [], []
    data, cfg_of = {}, {}
    for c in conds:
        cfg_of[c] = sel.get(c, {}).get("config", {}) if CONDITIONS[c]["grid"] != [{}] else {}
    for d in DOMAINS:
        runs = {}
        for c in conds:
            for s in CONFIRMATORY_SEEDS:
                rd = runs_dir / run_id(c, d, s, cfg_of[c])
                if not (rd / "DONE.json").exists():
                    (missing if c in required else missing_optional).append(rd.name)
                    continue
                runs[(c, s)] = load_run(rd)
        data[d] = DomainData(runs) if runs else None
    if missing:
        log("analyze", f"INCOMPLETE: {len(missing)} decision-relevant runs missing, e.g. {missing[:3]}")
        atomic_write_json(out / "incomplete.json", {"missing": missing, "missing_optional": missing_optional})
        return 2
    present = {c for c in conds if all((c, s) in data[d].stats for d in DOMAINS for s in CONFIRMATORY_SEEDS)}
    conds = [c for c in conds if c in present]
    if missing_optional:
        log("analyze", f"note: {len(missing_optional)} runs of reference/secondary conditions missing; "
                       f"they are left out of the summary: {sorted({m.split('__')[0] for m in missing_optional})}")
    # sanity checks on the ARTS joins and the clustering, before any statistic is computed
    for d in DOMAINS:
        st = data[d]
        for key, v in st.stats.items():
            n_joint = int(v["J"][1].sum())
            if expected_joint.get(d) is not None and n_joint != expected_joint[d]:
                raise SystemExit(f"ERROR {d} {key}: RT/RN joint set has {n_joint} sources, expected {expected_joint[d]}")
        if "" in st.k or st.K < args.min_clusters:
            raise SystemExit(f"ERROR {d}: {st.K} clusters (or empty cluster ids); the cluster bootstrap would be invalid")
    rng = np.random.default_rng(args.bootstrap_seed)
    R = args.replicates
    weights = {}
    for d in DOMAINS:
        K = data[d].K
        draws = rng.integers(0, K, size=(R, K))
        Wm = np.zeros((R + 1, K))
        Wm[0] = 1.0                                   # row 0: the observed data
        for r in range(R):
            Wm[r + 1] = np.bincount(draws[r], minlength=K)
        weights[d] = Wm
    # endpoint arrays: E[c][d][seed] -> vector over (observed + replicates)
    E = {c: {d: {s: data[d].endpoints((c, s), weights[d]) for s in CONFIRMATORY_SEEDS} for d in DOMAINS} for c in conds}

    def cond_mean(c, d, ep):            # mean over seeds -> vector
        return np.mean([E[c][d][s][ep] for s in CONFIRMATORY_SEEDS], axis=0)

    def pooled(c, ep):
        return np.mean([cond_mean(c, d, ep) for d in DOMAINS], axis=0)

    summary = {c: {ep: {"pooled": float(pooled(c, ep)[0]), **{d: float(cond_mean(c, d, ep)[0]) for d in DOMAINS}}
                   for ep in ("W", "J", "C") + FAMS} for c in conds}
    strongest = {ep: max(CORE, key=lambda c: summary[c][ep]["pooled"]) for ep in ("W", "J")}

    def ci(v):
        return [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]

    def evaluate(m: str) -> dict:
        res = {"method": m, "criteria": {}, "effects": {}}
        for ep in ("W", "J"):
            comp = strongest[ep]
            delta = pooled(m, ep) - pooled(comp, ep)
            per_dom = {d: float(cond_mean(m, d, ep)[0] - cond_mean(comp, d, ep)[0]) for d in DOMAINS}
            res["effects"][ep] = {"vs": comp, "pooled": float(delta[0]), "ci95": ci(delta), "per_domain": per_dom}
        dW, dJ = res["effects"]["W"], res["effects"]["J"]
        clean_delta = float(pooled(m, "C")[0] - pooled("Mixed", "C")[0])
        seed_ok = 0
        seed_rows = []
        for s in CONFIRMATORY_SEEDS:
            w = np.mean([E[m][d][s]["W"][0] - E[strongest["W"]][d][s]["W"][0] for d in DOMAINS])
            j = np.mean([E[m][d][s]["J"][0] - E[strongest["J"]][d][s]["J"][0] for d in DOMAINS])
            seed_rows.append({"seed": s, "dW": float(w), "dJ": float(j)})
            seed_ok += int(w > 0 and j > 0)
        res["seed_effects"] = seed_rows
        res["clean_minus_mixed"] = clean_delta
        res["criteria"] = {
            "A_worst_family": dW["pooled"] >= 2.0 and dW["ci95"][0] > 0,
            "B_joint_correctness": dJ["pooled"] >= 2.0 and dJ["ci95"][0] > 0,
            "C_domain_consistency": all(v >= -1.0 for v in list(dW["per_domain"].values()) + list(dJ["per_domain"].values())),
            "D_clean_guardrail": clean_delta >= -1.0,
            "E_seed_direction": seed_ok >= 2,
        }
        res["passes"] = all(res["criteria"].values())
        return res

    scct, paired = evaluate("SCCT"), evaluate("Paired")
    add = {ep: pooled("SCCT", ep) - pooled("Paired", ep) for ep in ("W", "J")}
    scct_adds = {ep: {"pooled": float(add[ep][0]), "ci95": ci(add[ep])} for ep in ("W", "J")}
    improves = scct_adds["W"]["pooled"] > 0 and scct_adds["J"]["pooled"] > 0
    if paired["passes"] and not improves:
        decision = "PAIRED_DATA_CANDIDATE: paired data beat the core comparators; the SCCT losses add nothing further"
    elif scct["passes"]:
        decision = "SCCT_PASS: proceed to the multi-encoder confirmation (Stage 2)"
    elif paired["passes"]:
        decision = "PAIRED_DATA_CANDIDATE: SCCT fails the rule but the paired-data construction passes"
    else:
        decision = "STOP_METHOD_PATH: neither SCCT nor paired data pass; continue as an analysis/theory paper"
    report = {"decision": decision, "strongest_comparator": strongest, "scct": scct, "paired": paired,
              "scct_minus_paired": scct_adds, "summary": summary, "selected_configs": cfg_of,
              "bootstrap": {"replicates": R, "seed": args.bootstrap_seed, "unit": "ARTS sentence cluster, within domain"}}
    atomic_write_json(out / "STAGE1_DECISION.json", report)
    md = ["# Stage-1 decision (protocol v1.0)", "", f"**{decision}**", "",
          f"Strongest comparator: W = {strongest['W']}, J = {strongest['J']}", ""]
    for r in (scct, paired):
        md += [f"## {r['method']}: {'PASS' if r['passes'] else 'FAIL'}", "",
               f"- dW vs {r['effects']['W']['vs']}: {r['effects']['W']['pooled']:+.2f} pp, 95% CI "
               f"[{r['effects']['W']['ci95'][0]:+.2f}, {r['effects']['W']['ci95'][1]:+.2f}]; per domain "
               + ", ".join(f"{d} {v:+.2f}" for d, v in r['effects']['W']['per_domain'].items()),
               f"- dJ vs {r['effects']['J']['vs']}: {r['effects']['J']['pooled']:+.2f} pp, 95% CI "
               f"[{r['effects']['J']['ci95'][0]:+.2f}, {r['effects']['J']['ci95'][1]:+.2f}]; per domain "
               + ", ".join(f"{d} {v:+.2f}" for d, v in r['effects']['J']['per_domain'].items()),
               f"- clean accuracy minus Mixed: {r['clean_minus_mixed']:+.2f} pp",
               "- seeds (dW, dJ): " + "; ".join(f"{x['seed']}: {x['dW']:+.2f}, {x['dJ']:+.2f}" for x in r["seed_effects"]),
               "- criteria: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in r["criteria"].items()), ""]
    md += [f"SCCT minus Paired: dW {scct_adds['W']['pooled']:+.2f} pp, dJ {scct_adds['J']['pooled']:+.2f} pp", "",
           "## All conditions (mean of 3 seeds; equal-domain pooled)", "",
           "| Condition | W | J | Clean | RevTgt | RevNon | AddDiff |", "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {c} | {summary[c]['W']['pooled']:.2f} | {summary[c]['J']['pooled']:.2f} | {summary[c]['C']['pooled']:.2f} | "
           f"{summary[c]['RevTgt']['pooled']:.2f} | {summary[c]['RevNon']['pooled']:.2f} | {summary[c]['AddDiff']['pooled']:.2f} |"
           for c in conds]
    atomic_write_text(out / "STAGE1_DECISION.md", "\n".join(md) + "\n")
    log("analyze", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
