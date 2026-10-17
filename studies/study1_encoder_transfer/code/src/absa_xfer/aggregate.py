from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def aggregate_run_results(runs_root: str|Path, out_csv: str|Path, out_parquet: str|Path):
    rows=[]
    for result in Path(runs_root).rglob("run_result.json"):
        with open(result,encoding="utf-8") as f:
            obj=json.load(f)
        run_dir=result.parent
        # Optional run metadata written by orchestrator.
        meta={}
        mp=run_dir/"run_meta.json"
        if mp.exists():
            meta=json.loads(mp.read_text(encoding="utf-8"))
        for eval_cond, metrics in obj.get("eval_summary",{}).items():
            for metric,value in metrics.items():
                rows.append({
                    **{k:meta.get(k) for k in (
                        "run_id","dataset","model","training_condition","perturbation_budget","seed"
                    )},
                    "evaluation_condition":eval_cond,
                    "metric":metric,"value":value,
                    "run_path":str(run_dir)
                })
    df=pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_csv,index=False)
    try:
        df.to_parquet(out_parquet,index=False)
    except Exception as e:
        print("Parquet write skipped:",e)
    return df
