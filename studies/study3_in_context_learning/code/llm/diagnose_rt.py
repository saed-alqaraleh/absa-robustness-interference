#!/usr/bin/env python3
"""Error anatomy on ARTS RevTgt (and RevNon): what do systems predict when they fail?

For every system (each complete LLM zero-shot and with Clean demonstrations; fine-tuned RoBERTa
CleanMatched / Mixed / Paired from Stage 1, all seeds pooled):
  * persistence  = share of RevTgt errors that repeat the ORIGINAL (pre-edit) label of the source
                   -> the edit that should flip the label was ignored (sensitivity failure)
  * neutral      = share of RevTgt errors predicting neutral
  * on RevNon errors: share predicting the label the non-target edit would suggest is not knowable
    here, so RevNon errors are split into neutral / other only.
Items that every system gets wrong are written out for a manual label audit (possible ARTS noise).
Descriptive only; equal-domain means. Reads frozen files; writes <out>/analysis/RT_ERROR_ANATOMY.*
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.common import DEMO_SEEDS, DOMAINS, OUT_SUBDIR, atomic_write_json, atomic_write_text, job_id, jobs, log, read_jsonl  # noqa: E402

REF_CONDS = ("CleanMatched", "Mixed", "Paired")


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def systems(root: Path, out_root: Path, d: str) -> dict[str, dict[str, list[list[dict]]]]:
    """name -> set -> list of runs (each run = rows)."""
    out = {}
    ids = [job_id(j) for j in jobs(["x"])]
    for mdir in sorted(p for p in out_root.iterdir() if p.is_dir() and any(p.glob("*.DONE.json"))):
        if not all((mdir / f"{j}.DONE.json").exists() for j in ids):
            continue
        short = mdir.name.split("__")[-1]
        for cond, seeds in (("ZS", (0,)), ("Clean", DEMO_SEEDS)):
            runs = [read_csv(mdir / f"{job_id({'domain': d, 'condition': cond, 'seed': s})}.csv") for s in seeds]
            out[f"{short} [{cond}]"] = {st: [[r for r in rr if r["set"] == st] for rr in runs] for st in ("RevTgt", "RevNon")}
    confirm = root / "stage1_v1" / "runs" / "confirm"
    for c in REF_CONDS:
        hits = [h for s in DEMO_SEEDS for h in confirm.glob(f"{c}__{d}__*__seed{s}") if (h / "DONE.json").exists()]
        if len(hits) == len(DEMO_SEEDS):
            out[f"RoBERTa-ft [{c}]"] = {st: [read_csv(h / f"predictions_{st}.csv") for h in hits] for st in ("RevTgt", "RevNon")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-subdir", default=OUT_SUBDIR)
    args = ap.parse_args(argv)
    root = args.root.expanduser().resolve()
    out = root / args.out_subdir / "analysis"
    res = {"by_domain": {}, "universal_failures": {}}
    audit = []
    per = defaultdict(lambda: defaultdict(list))
    for d in DOMAINS:
        test = {r["source_instance_id"]: r for r in read_jsonl(root / "data_processed" / d / "test.jsonl")}
        arts = {f: {r["source_instance_id"]: r for r in read_jsonl(root / "data_processed" / d / "evaluation" / f"{f}.jsonl")}
                for f in ("RevTgt", "RevNon")}
        cover = sum(s in test for s in arts["RevTgt"]) / max(1, len(arts["RevTgt"]))
        syst = systems(root, root / args.out_subdir, d)
        res["by_domain"][d] = {"rt_items_with_original": cover, "systems": {}}
        wrong_all = {f: None for f in arts}
        for name, sets in syst.items():
            row = {}
            for f in ("RevTgt", "RevNon"):
                errs = Counter()
                n = n_err = 0
                wrong_any_run = Counter()
                for rr in sets[f]:
                    for r in rr:
                        n += 1
                        if int(r["correct"]):
                            continue
                        n_err += 1
                        wrong_any_run[r["source_instance_id"]] += 1
                        orig = test.get(r["source_instance_id"], {}).get("label")
                        if f == "RevTgt" and r["pred"] == orig:
                            errs["persist"] += 1
                        elif r["pred"] == "neutral":
                            errs["neutral"] += 1
                        else:
                            errs["other"] += 1
                runs = len(sets[f])
                majority_wrong = {s for s, k in wrong_any_run.items() if k * 2 > runs}
                wrong_all[f] = majority_wrong if wrong_all[f] is None else wrong_all[f] & majority_wrong
                row[f] = {"accuracy": 100 * (1 - n_err / max(1, n)), "errors": n_err,
                          **{k: 100 * errs[k] / max(1, n_err) for k in ("persist", "neutral", "other")}}
                for k, v in row[f].items():
                    per[(name, f)][k].append(v)
            res["by_domain"][d]["systems"][name] = row
        for f, ids in wrong_all.items():
            ids = ids or set()
            res["universal_failures"][f"{d}/{f}"] = {"n": len(ids), "of": len(arts[f]), "share": 100 * len(ids) / max(1, len(arts[f]))}
            for s in sorted(ids):
                a, t = arts[f][s], test.get(s, {})
                audit.append({"domain": d, "family": f, "source_instance_id": s, "aspect": a["aspect_term"],
                              "edited_sentence": a["sentence"], "gold": a["label"],
                              "original_sentence": t.get("sentence", ""), "original_label": t.get("label", "")})
    res["pooled"] = {f"{n} | {f}": {k: sum(v) / len(v) for k, v in kv.items()} for (n, f), kv in per.items()}
    atomic_write_json(out / "RT_ERROR_ANATOMY.json", res)
    if audit:
        with (out / "UNIVERSAL_FAILURES_for_audit.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(audit[0]))
            w.writeheader()
            w.writerows(audit)
    L = ["# Error anatomy on ARTS (equal-domain means, %)", "",
         "| System | RevTgt acc | RevTgt errors: keeps original label | → neutral | other | RevNon acc | RevNon errors: → neutral |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    names = sorted({n for n, _ in per})
    for n in names:
        rt, rn = res["pooled"][f"{n} | RevTgt"], res["pooled"][f"{n} | RevNon"]
        L.append(f"| {n} | {rt['accuracy']:.1f} | {rt['persist']:.1f} | {rt['neutral']:.1f} | {rt['other']:.1f} | "
                 f"{rn['accuracy']:.1f} | {rn['neutral']:.1f} |")
    L += ["", "Items wrong for every system (seed-majority for multi-run systems):", ""]
    for k, v in res["universal_failures"].items():
        L.append(f"- {k}: {v['n']} of {v['of']} ({v['share']:.1f}%)")
    L += ["", "RevTgt items with a matching original clean item: "
          + ", ".join(f"{d} {res['by_domain'][d]['rt_items_with_original']:.3f}" for d in DOMAINS),
          "", "Audit file: UNIVERSAL_FAILURES_for_audit.csv (check whether the gold label of each edited item is correct).", ""]
    md = "\n".join(L)
    atomic_write_text(out / "RT_ERROR_ANATOMY.md", md)
    print(md)
    log("diagnose", f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
