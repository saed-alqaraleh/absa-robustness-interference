#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from absa_xfer.io_utils import read_jsonl, atomic_write_json
from absa_xfer.schemas import AspectExample,OpinionRecord,OpinionSpan
from absa_xfer.transformations import build_aspect_expression_library
from absa_xfer.source_control import audit_eligibility, select_common_pool, audit_selected_pool
from absa_xfer.hashing import hash_object, sha256_file


def load_examples(path):
    return [AspectExample(**{k:r[k] for k in AspectExample.__dataclass_fields__}) for r in read_jsonl(path)]

def load_opinions(path):
    out={}
    for r in read_jsonl(path):
        spans=tuple(OpinionSpan(**s) for s in r["spans"])
        rec=OpinionRecord(r["source_instance_id"],r["cluster_id"],spans,r["matched_by"],r["provenance"])
        out[rec.source_instance_id]=rec
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--opinions",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--budget",type=int,default=None)
    ap.add_argument("--seed",type=int,default=20260913)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    ex=load_examples(args.train); op=load_opinions(args.opinions)
    lib=build_aspect_expression_library(ex,op)
    rows,counts=audit_eligibility(ex,op,lib)
    erows=out/"eligibility_rows.csv"; lpath=out/"train_only_aspect_expression_library.csv"
    pd.DataFrame(rows).to_csv(erows,index=False)
    pd.DataFrame([x.__dict__ for x in lib]).to_csv(lpath,index=False)
    common_n=sum(r["eligible_all3"] for r in rows)
    budget=args.budget if args.budget is not None else common_n
    if budget > common_n:
        raise ValueError(f"requested budget {budget} exceeds common eligible pool {common_n}")
    if budget <= 0:
        raise ValueError("common eligible source pool is empty; source-composition gate FAIL")
    selected=select_common_pool(rows,budget,args.seed)
    (out/"selected_source_ids.txt").write_text("\n".join(selected)+"\n",encoding="utf-8")
    selection={
        "selection_policy":"common_eligibility_pool",
        "selection_seed":args.seed,
        "budget":budget,
        "selected_ids":selected,
        "selected_ids_hash":hash_object(selected),
        "input_train_sha256":sha256_file(args.train),
        "input_opinions_sha256":sha256_file(args.opinions),
    }
    atomic_write_json(out/"source_selection.json",selection)
    summary={"eligibility_counts":counts,"common_pool_n":common_n,"budget":budget,
             "selected_pool":audit_selected_pool(rows,set(selected)),
             "source_selection_hash":hash_object(selection)}
    atomic_write_json(out/"eligibility_summary.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
