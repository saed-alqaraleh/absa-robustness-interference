#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from absa_xfer.io_utils import read_jsonl, write_jsonl, atomic_write_json
from absa_xfer.schemas import AspectExample
from absa_xfer.towe import load_towe_tsv, align_towe_to_examples

def exs(rows):
    return [AspectExample(**{k:r[k] for k in AspectExample.__dataclass_fields__}) for r in rows]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--examples",required=True)
    ap.add_argument("--towe-tsv",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    examples=exs(read_jsonl(args.examples))
    trows=load_towe_tsv(args.towe_tsv)
    aligned,report=align_towe_to_examples(examples,trows,provenance=args.towe_tsv)
    write_jsonl(out/"opinion_records.jsonl",[v.to_dict() for v in aligned.values()])
    write_jsonl(out/"opinion_alignment_report.jsonl",report)
    summary={
      "examples":len(examples),"matched":sum(r["status"]=="matched" for r in report),
      "unmatched":sum(r["status"]=="unmatched" for r in report),
      "ambiguous":sum(r["status"]=="ambiguous" for r in report)
    }
    atomic_write_json(out/"opinion_alignment_summary.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
