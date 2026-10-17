#!/usr/bin/env python3
"""Stage 1, step 1 (CPU): build every Stage-1 training file and the pre-run audit.

Implements protocol v1.0, Decision 3 and the "Required pre-run audit":

* common source pool per domain: Study-I common pool sources whose frozen RevTgt,
  reversal-only RevNon and AddDiff derivatives exist and for which a target-opinion
  intensification (TgtInt) can be generated at the span RevTgt edits;
* Stage-1 budget B1 = 6 * floor(min(B_D, 2|P_D|) / 6);
* added records for every condition (Mixed, BAT, Fei, triplet = Paired/SCCT,
  Clean-Matched reference); the clean base set is the frozen post-firewall train split;
* the robustness-dev set (RevTgt, reversal-only RevNon, AddDiff generated from the dev
  split with the verified generator), used only for tuning;
* a machine-readable manifest with pool sizes, B1, source ids per condition, SHA-256 of
  every generated file, code hashes and the protocol digest.

Frozen inputs are only read. Outputs go to <root>/stage1_v1/data. A success marker is
written only when every validity check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.common import (  # noqa: E402
    ARTS_FAMILIES, CONSTRUCTION_KEY, DEGREE_ADVERBS, DOMAINS, GENERATOR_SEED, HISTORICAL_BUDGET, KIT, MIN_TUNING_ITEMS,
    RESERVE_FRACTION,
    POLAR_FLIP, PROTOCOL_SHA256, STAGE1_SUBDIR, amendments, atomic_write_json, atomic_write_text, check_protocol, log,
    ordered, read_jsonl, sha256_file, sha256_text, stage1_budget, write_jsonl,
)

FAMILY_FILES = {"RT": "Clean+RevTgt", "RN": "Clean+RevNon", "AD": "Clean+AddDiff"}
TOWE = {"restaurant": "14res", "laptop": "14lap"}


def deterministic_choice(items, key: str) -> str:
    """Identical to absa_xfer.transformations.deterministic_choice (frozen generator)."""
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return sorted(items)[h % len(items)]


def revnon_reversal_only(rec: dict) -> bool:
    ops = [o["operation"] for o in rec.get("generation", {}).get("metadata", {}).get("operations", [])]
    return bool(ops) and not any(o.startswith("intensify") for o in ops)


def sub_operator(op: str) -> str:
    return "antonym" if op.startswith("antonym") else ("negation" if "negation" in op else "other")


def make_tgtint(rt: dict) -> dict | None:
    """Target-opinion intensification at the exact span RevTgt edits; label unchanged."""
    g = rt.get("generation", {})
    span = g.get("metadata", {}).get("opinion_span")
    sent, sid = g.get("original_sentence"), rt["source_instance_id"]
    if not span or sent is None or g.get("original_label") not in POLAR_FLIP:
        return None
    s, e = int(span[0]), int(span[1])
    adv = deterministic_choice(list(DEGREE_ADVERBS), f"{GENERATOR_SEED}|{sid}|TgtInt|adv")
    return {"sentence": sent[:s] + f"{adv} {sent[s:e]}" + sent[e:], "operator": f"intensify:{adv}",
            "opinion_span": [s, e]}


def added(base: dict, fam: str, sentence: str, label: str, operator: str) -> dict:
    return {"sentence": sentence, "aspect_term": base["aspect_term"], "label": label,
            "orig_label": base["label"], "source_instance_id": base["source_instance_id"],
            "cluster_id": base["cluster_id"], "is_augmented": True, "stage1_family": fam,
            "operator": operator, "group": base["source_instance_id"]}


def build_domain(root: Path, d: str, out: Path, reserve_fraction: float = 0.0) -> dict:
    train = read_jsonl(root / "data_processed" / d / "train.jsonl")
    by_id = {r["source_instance_id"]: r for r in train}
    sel = json.loads((root / "audits" / "eligibility" / d /
                                    "FINAL_GENERATION_VALID_SOURCE_SELECTION.json").read_text(encoding="utf-8"))
    study1_pool = list(sel["selected_ids"] if isinstance(sel, dict) else sel)
    fam_recs: dict[str, dict[str, dict]] = {}
    for fam, cond in FAMILY_FILES.items():
        recs = [r for r in read_jsonl(root / "transformations" / "materialized" / d / cond / "train.jsonl")
                if r.get("is_augmented")]
        fam_recs[fam] = {}
        for r in recs:
            fam_recs[fam].setdefault(r["source_instance_id"], r)
    reasons, pool, ti = Counter(), [], {}
    for sid in study1_pool:
        if sid not in by_id:
            reasons["not_in_frozen_train"] += 1
            continue
        if sid not in fam_recs["RT"]:
            reasons["no_RT"] += 1
            continue
        if sid not in fam_recs["AD"]:
            reasons["no_AD"] += 1
            continue
        if sid not in fam_recs["RN"] or not revnon_reversal_only(fam_recs["RN"][sid]):
            reasons["no_reversal_only_RN"] += 1
            continue
        t = make_tgtint(fam_recs["RT"][sid])
        if t is None:
            reasons["no_TgtInt"] += 1
            continue
        ti[sid] = t
        pool.append(sid)
    full_pool = list(pool)
    reserve = ordered(full_pool, "tuning-reserve")[:math.ceil(reserve_fraction * len(full_pool))] if reserve_fraction else []
    pool = [x for x in full_pool if x not in set(reserve)]
    B1 = stage1_budget(HISTORICAL_BUDGET[d], len(pool))
    third, half = B1 // 3, B1 // 2

    def rec(fam: str, sid: str) -> dict:
        base = by_id[sid]
        if fam == "TI":
            return added(base, "TI", ti[sid]["sentence"], base["label"], ti[sid]["operator"])
        r = fam_recs[fam][sid]
        op = r["generation"]["metadata"].get("operation") if fam == "RT" else \
            ";".join(o["operation"] for o in r["generation"]["metadata"].get("operations", [])) if fam == "RN" else "adddiff"
        return added(base, fam, r["sentence"], r["label"], op)

    common = ordered(pool, "paired")
    constructions = {
        "triplet": [(f, s) for s in common[:third] for f in ("RT", "RN", "AD")],
        "bat": [(f, s) for s in common[:half] for f in ("TI", "RT")],
        "fei": [(f, s) for s in common[:third] for f in ("TI", "RT", "AD")],
        "mixed": [(f, s) for f in ("RT", "RN", "AD") for s in ordered(pool, f"mixed|{f}")[:third]],
    }
    cm_order = ordered(pool, "cleanmatched")
    clean_base = [{**r, "is_augmented": False, "stage1_family": "Clean", "orig_label": r["label"],
                   "operator": "none", "group": r["source_instance_id"]} for r in train]
    manifest: dict = {"domain": d, "study1_pool_size": len(study1_pool), "eligible_pool_size": len(full_pool),
                      "tuning_reserve": sorted(reserve), "pool_size": len(pool),
                      "pool_exclusions": dict(reasons), "pool_sha256": sha256_text("\n".join(sorted(pool))),
                      "historical_budget": HISTORICAL_BUDGET[d], "B1": B1, "conditions": {}}
    checks: list[tuple[str, bool]] = []
    for name, pairs in constructions.items():
        rows = [rec(f, s) for f, s in pairs]
        path = out / d / name / "train.jsonl"
        write_jsonl(path, clean_base + rows)
        fams = Counter(r["stage1_family"] for r in rows)
        manifest["conditions"][name] = {
            "added_records": len(rows), "family_counts": dict(fams),
            "sources": sorted({s for _, s in pairs}), "n_sources": len({s for _, s in pairs}),
            "train_file": str(path.relative_to(out.parent)), "sha256": sha256_file(path)}
        checks.append((f"{d}/{name}: exactly B1 added records", len(rows) == B1))
        checks.append((f"{d}/{name}: no duplicate (family, source)", len(set(pairs)) == len(pairs)))
        checks.append((f"{d}/{name}: sources inside the common pool", {s for _, s in pairs} <= set(pool)))
        for r in rows:
            want = POLAR_FLIP[r["orig_label"]] if r["stage1_family"] == "RT" else r["orig_label"]
            if r["label"] != want:
                checks.append((f"{d}/{name}: label rule for {r['stage1_family']} {r['source_instance_id']}", False))
                break
        else:
            checks.append((f"{d}/{name}: gold labels follow the edit (RT flips; RN, AD, TI keep)", True))
    # Clean-Matched reference: B1 clean repeats cycling through the pool
    reps = [cm_order[i % len(cm_order)] for i in range(B1)]
    cm_rows = [{**by_id[s], "is_augmented": True, "stage1_family": "CleanRepeat", "orig_label": by_id[s]["label"],
                "operator": "repeat", "group": s} for s in reps]
    path = out / d / "clean_matched" / "train.jsonl"
    write_jsonl(path, clean_base + cm_rows)
    manifest["conditions"]["clean_matched"] = {"added_records": len(cm_rows), "family_counts": {"CleanRepeat": len(cm_rows)},
                                               "sources": sorted(set(reps)), "n_sources": len(set(reps)),
                                               "train_file": str(path.relative_to(out.parent)), "sha256": sha256_file(path)}
    checks.append((f"{d}/clean_matched: exactly B1 added records", len(cm_rows) == B1))
    trip = constructions["triplet"]
    agree = Counter()
    for s in common[:third]:
        rt_op = fam_recs["RT"][s]["generation"]["metadata"].get("operation", "")
        rn_subs = {sub_operator(o["operation"]) for o in fam_recs["RN"][s]["generation"]["metadata"].get("operations", [])}
        agree["same_sub_operator" if sub_operator(rt_op) in rn_subs else "different_sub_operator"] += 1
    manifest["triplet_sub_operator_agreement"] = dict(agree)
    first_words = ("not", "never", "no", "n't", "very", "really", "extremely", "too", "so", "quite", "pretty")
    odd = [s for s in pool if (fam_recs["RT"][s]["generation"]["original_sentence"][slice(*ti[s]["opinion_span"])]
                               .lower().split() or [""])[0] in first_words]
    manifest["tgtint_span_starts_with_negator_or_degree_word"] = {"n": len(odd), "share_of_pool": len(odd) / max(1, len(pool))}
    checks.append((f"{d}/triplet: complete RT+RN+AD per source",
                   all(c == 3 for c in Counter(s for _, s in trip).values())))
    checks.append((f"{d}: B1 = 6*floor(min(B, 2|P|)/6)", B1 == stage1_budget(HISTORICAL_BUDGET[d], len(pool)) and B1 % 6 == 0))
    manifest["checks"] = [{"check": c, "pass": ok} for c, ok in checks]
    # frozen derivatives of reserved sources: no Stage-1 construction ever trains on them
    manifest["_reserve_rows"] = {
        fam: [{"sentence": (rec(fam, x)["sentence"]), "aspect_term": by_id[x]["aspect_term"], "label": rec(fam, x)["label"],
               "orig_label": by_id[x]["label"], "source_instance_id": x, "cluster_id": by_id[x]["cluster_id"],
               "stage1_family": fam, "origin": "reserved_pool"} for x in sorted(reserve)]
        for fam in ("RT", "RN", "AD")}
    return manifest


def load_package(code_src: Path):
    sys.path.insert(0, str(code_src))
    import absa_xfer.transformations as T  # noqa: E402
    from absa_xfer.schemas import AspectExample  # noqa: E402
    return T, AspectExample


def load_opinions_with_script(code_dir: Path, path: Path):
    spec = importlib.util.spec_from_file_location("s04", code_dir / "scripts" / "04_materialize_conditions.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m.load_examples, m.load_opinions(str(path))


def write_tuning_set(out: Path, d: str, dev_fams: dict[str, list[dict]], reserve_rows: dict[str, list[dict]], info: dict) -> dict:
    """Write the tuning set (dev-split derivatives + optional reserved-pool derivatives) and check adequacy."""
    fams = {f: list(dev_fams.get(f, [])) + list(reserve_rows.get(f, [])) for f in ("RT", "RN", "AD")}
    info["files"] = {}
    for fam, rows in fams.items():
        p = out / d / "robust_dev" / f"{fam}.jsonl"
        write_jsonl(p, rows)
        info["files"][fam] = {"n": len(rows), "from_dev_split": len(dev_fams.get(fam, [])),
                              "from_reserved_pool": len(reserve_rows.get(fam, [])), "sha256": sha256_file(p)}
    both = {r["source_instance_id"] for r in fams["RT"]} & {r["source_instance_id"] for r in fams["RN"]}
    info["rt_rn_joint_sources"] = len(both)
    info["adequate"] = len(fams["RN"]) >= MIN_TUNING_ITEMS and len(both) >= MIN_TUNING_ITEMS and len(fams["RT"]) > 0 \
        and len(fams["AD"]) > 0
    if not info["adequate"]:
        log("data", f"{d}: tuning set too small (RN {len(fams['RN'])}, RT&RN sources {len(both)}; need >= {MIN_TUNING_ITEMS}). "
                    "W_dev/J_dev would be undefined or arbitrary. See IMPLEMENTATION_NOTES N10 (amendment v1.1).")
    return info


def build_robust_dev(root: Path, d: str, code_dir: Path, out: Path) -> dict:
    """Generate RevTgt / reversal-only RevNon / AddDiff from the dev split (tuning only)."""
    T, _ = load_package(code_dir / "src")
    from nltk.corpus import wordnet as wn  # the generator needs WordNet 3.0
    if wn.get_version() != "3.0":
        raise SystemExit(f"WordNet 3.0 required, found {wn.get_version()}")
    dev_path = root / "data_processed" / d / "dev.jsonl"
    towe = root / "data_processed" / "towe_validated" / d / "train_validated.tsv"
    ops_dir = out / d / "robust_dev" / "dev_opinions"
    env = dict(os.environ, PYTHONPATH=str(code_dir / "src"))
    rc = subprocess.run([sys.executable, "scripts/02_align_towe.py", "--examples", str(dev_path), "--towe-tsv", str(towe),
                         "--out-dir", str(ops_dir)], cwd=str(code_dir), env=env, capture_output=True, text=True)
    if rc.returncode != 0:
        raise SystemExit(f"dev opinion alignment failed: {rc.stdout[-300:]} {rc.stderr[-300:]}")
    load_examples, dev_ops = load_opinions_with_script(code_dir, ops_dir / "opinion_records.jsonl")
    _, train_ops = load_opinions_with_script(code_dir, root / "transformations" / "opinion_alignment" / d / "train" / "opinion_records.jsonl")
    dev_ex = load_examples(str(dev_path))
    train_ex = load_examples(str(root / "data_processed" / d / "train.jsonl"))
    library = T.build_aspect_expression_library(train_ex, train_ops)
    by_cluster = defaultdict(list)
    for x in dev_ex:
        by_cluster[x.cluster_id].append(x)
    fams: dict[str, list[dict]] = {"RT": [], "RN": [], "AD": []}
    reasons = Counter()
    for x in dev_ex:
        for fam, fn in (("RT", lambda: T.generate_revtgt(x, by_cluster[x.cluster_id], dev_ops, GENERATOR_SEED)),
                        ("RN", lambda: T.generate_revnon(x, by_cluster[x.cluster_id], dev_ops, GENERATOR_SEED)),
                        ("AD", lambda: T.generate_adddiff(x, by_cluster[x.cluster_id], dev_ops, library, GENERATOR_SEED))):
            try:
                y = fn()
            except ValueError as e:
                reasons[f"{fam}:{e}"] += 1
                continue
            g = y.to_dict()
            if fam == "RN" and not revnon_reversal_only({"generation": g}):
                reasons["RN:not_reversal_only"] += 1
                continue
            fams[fam].append({"sentence": y.transformed_sentence, "aspect_term": y.transformed_aspect,
                              "label": y.transformed_label, "orig_label": x.label,
                              "source_instance_id": x.source_instance_id, "cluster_id": x.cluster_id,
                              "stage1_family": fam, "generation": g})
    info = {"dev_examples": len(dev_ex), "dev_opinions": len(dev_ops), "exclusions": dict(reasons), "_fams": fams}
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--code-dir", type=Path, default=None,
                    help="Generator code (default: <root>/release_v2/08_Code_and_Regeneration_v2/code)")
    ap.add_argument("--skip-robust-dev", action="store_true", help="skip the dev-split derivatives (offline tests only)")
    ap.add_argument("--tuning-set", choices=("dev", "dev+reserved"), default=None,
                    help="dev = protocol v1.0 (dev-split derivatives only); dev+reserved = amendment v1.1 option, "
                         f"adds the frozen derivatives of {int(RESERVE_FRACTION * 100)}%% of the pool held out of every construction")
    args = ap.parse_args()
    check_protocol()
    args.tuning_set = args.tuning_set or amendments()["tuning_set"]
    root = args.root.expanduser().resolve()
    code_dir = (args.code_dir or root / "release_v2" / "08_Code_and_Regeneration_v2" / "code").expanduser().resolve()
    out = root / STAGE1_SUBDIR / "data"
    marker = out / "_DATA_READY.json"
    if marker.exists():
        marker.unlink()
    manifest = {"protocol_sha256": PROTOCOL_SHA256, "amendments": amendments(), "construction_key": CONSTRUCTION_KEY,
                "tuning_set": args.tuning_set,
                "generator_seed": GENERATOR_SEED, "generator_code_dir": str(code_dir),
                "generator_code_sha256": {f: sha256_file(code_dir / "src" / "absa_xfer" / f)
                                          for f in ("transformations.py", "towe.py")
                                          if (code_dir / "src" / "absa_xfer" / f).is_file()},
                "kit_sha256": {str(p.relative_to(KIT)): sha256_file(p) for p in sorted((KIT / "stage1").glob("*.py"))},
                "frozen_inputs_sha256": {}, "domains": {}}
    for d in DOMAINS:
        for rel in [f"data_processed/{d}/train.jsonl", f"data_processed/{d}/dev.jsonl", f"data_processed/{d}/test.jsonl"] + \
                   [f"data_processed/{d}/evaluation/{f}.jsonl" for f in ARTS_FAMILIES] + \
                   [f"transformations/materialized/{d}/{c}/train.jsonl" for c in FAMILY_FILES.values()] + \
                   [f"audits/eligibility/{d}/FINAL_GENERATION_VALID_SOURCE_SELECTION.json"]:
            manifest["frozen_inputs_sha256"][rel] = sha256_file(root / rel)
        m = build_domain(root, d, out, RESERVE_FRACTION if args.tuning_set == "dev+reserved" else 0.0)
        reserve_rows = m.pop("_reserve_rows")
        if args.skip_robust_dev and args.tuning_set == "dev":
            m["robust_dev"] = {"skipped": True}
        else:
            info = build_robust_dev(root, d, code_dir, out) if not args.skip_robust_dev else {"dev_split": "skipped", "_fams": {}}
            m["robust_dev"] = write_tuning_set(out, d, info.pop("_fams"), reserve_rows, info)
            m["checks"].append({"check": f"{d}/tuning set adequate (RN >= {MIN_TUNING_ITEMS} and RT&RN sources >= {MIN_TUNING_ITEMS})",
                                "pass": m["robust_dev"]["adequate"]})
        manifest["domains"][d] = m
        log("data", f"{d}: pool={m['pool_size']} (excluded {m['pool_exclusions']}), B1={m['B1']}, "
                    f"triplet sub-operator agreement {m['triplet_sub_operator_agreement']}")
    checks = [c for d in DOMAINS for c in manifest["domains"][d]["checks"]]
    ok = all(c["pass"] for c in checks)
    manifest["all_checks_pass"] = ok
    atomic_write_json(out / "PRE_RUN_AUDIT.json", manifest)
    lines = ["# Stage-1 pre-run audit", "", f"Protocol SHA-256: `{PROTOCOL_SHA256}`", "",
             f"**{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}** ({sum(c['pass'] for c in checks)}/{len(checks)})", "",
             "| Domain | Pool | B1 | Mixed RT/RN/AD | Triplet sources | BAT sources | Fei sources |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in DOMAINS:
        m = manifest["domains"][d]
        c = m["conditions"]
        lines.append(f"| {d} | {m['pool_size']} | {m['B1']} | {c['mixed']['family_counts']} | {c['triplet']['n_sources']} | "
                     f"{c['bat']['n_sources']} | {c['fei']['n_sources']} |")
    lines += ["", f"Tuning set: `{args.tuning_set}`", "", "| Domain | RT | RN | AD | RT&RN sources | Adequate |",
              "| --- | --- | --- | --- | --- | --- |"]
    for d in DOMAINS:
        rdv = manifest["domains"][d].get("robust_dev", {})
        if "files" in rdv:
            f = rdv["files"]
            lines.append(f"| {d} | {f['RT']['n']} | {f['RN']['n']} | {f['AD']['n']} | {rdv['rt_rn_joint_sources']} | {rdv['adequate']} |")
    lines += ["", "| Check | Result |", "| --- | --- |"] + [f"| {c['check']} | {'pass' if c['pass'] else 'FAIL'} |" for c in checks]
    atomic_write_text(out / "PRE_RUN_AUDIT.md", "\n".join(lines) + "\n")
    if ok:
        atomic_write_json(marker, {"protocol_sha256": PROTOCOL_SHA256,
                                   "audit_sha256": sha256_file(out / "PRE_RUN_AUDIT.json")})
    log("data", f"{'READY' if ok else 'FAILED'}; audit: {out / 'PRE_RUN_AUDIT.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
