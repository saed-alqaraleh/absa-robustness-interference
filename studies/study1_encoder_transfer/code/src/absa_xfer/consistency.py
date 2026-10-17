from __future__ import annotations
import csv, json
from pathlib import Path
from collections import Counter
import pandas as pd


def campaign_consistency(plan_csv: str | Path, results_csv: str | Path, runs_root: str | Path) -> dict:
    with open(plan_csv, newline="", encoding="utf-8") as f:
        plan=list(csv.DictReader(f))
    expected={r["run_id"] for r in plan}
    done=set()
    bad_done=[]
    for p in Path(runs_root).rglob("DONE.json"):
        try:
            obj=json.loads(p.read_text(encoding="utf-8"))
            rid=(p.parent/"run_meta.json")
            meta=json.loads(rid.read_text(encoding="utf-8")) if rid.exists() else {}
            if obj.get("status")=="complete" and meta.get("run_id"):
                done.add(meta["run_id"])
            else:
                bad_done.append(str(p))
        except Exception:
            bad_done.append(str(p))
    duplicates=[]
    result_runs=set()
    result_rows=0
    if Path(results_csv).exists():
        df=pd.read_csv(results_csv)
        result_rows=len(df)
        if "run_id" in df:
            result_runs=set(df["run_id"].dropna().astype(str))
            key_cols=[c for c in ("run_id","evaluation_condition","metric") if c in df]
            if key_cols:
                counts=df.groupby(key_cols, dropna=False).size()
                duplicates=[{"key":list(k if isinstance(k,tuple) else (k,)),"n":int(v)} for k,v in counts[counts>1].items()]
    return {
        "expected_runs":len(expected),
        "done_runs":len(done),
        "missing_done":sorted(expected-done),
        "unexpected_done":sorted(done-expected),
        "results_rows":result_rows,
        "runs_missing_from_results":sorted(done-result_runs),
        "duplicate_result_keys":duplicates,
        "malformed_done_files":bad_done,
        "complete": expected == done and not duplicates and not bad_done,
    }
