#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from absa_xfer.io_utils import read_jsonl,atomic_write_json,atomic_write_text
from absa_xfer.integrity import integrity_report


def to_markdown(rep: dict) -> str:
    lines=["# DATA INTEGRITY REPORT","",f"**Blocking:** `{rep['blocking']}`","","## Counts","",
           "| Split | Records |","|---|---:|"]
    for k,v in rep["counts"].items(): lines.append(f"| {k} | {v} |")
    lines += ["","## Label distributions",""]
    for split,d in rep["label_distribution"].items(): lines.append(f"- **{split}:** {json.dumps(d,sort_keys=True)}")
    lines += ["",f"Duplicate normalized sentence excess count: **{rep['duplicate_normalized_sentences_total']}**","","## Issues",""]
    if not rep["issues"]: lines.append("No integrity issues detected by the implemented checks.")
    else:
        lines += ["| Severity | Type | N |","|---|---|---:|"]
        for x in rep["issues"]: lines.append(f"| {x['severity']} | {x['type']} | {x['n']} |")
    return "\n".join(lines)+"\n"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True);ap.add_argument("--dev",required=True);ap.add_argument("--test",required=True)
    ap.add_argument("--derivatives",default=None);ap.add_argument("--out",required=True);ap.add_argument("--markdown",default=None)
    a=ap.parse_args(); tr=read_jsonl(a.train);dv=read_jsonl(a.dev);te=read_jsonl(a.test);der=read_jsonl(a.derivatives) if a.derivatives else []
    rep=integrity_report(tr,dv,te,der); atomic_write_json(a.out,rep)
    if a.markdown: atomic_write_text(a.markdown,to_markdown(rep))
    print(json.dumps(rep,indent=2))
    if rep["blocking"]: sys.exit(3)
if __name__=="__main__": main()
