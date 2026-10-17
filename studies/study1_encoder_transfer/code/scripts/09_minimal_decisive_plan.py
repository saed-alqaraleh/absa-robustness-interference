#!/usr/bin/env python
"""Backward-compatible small planner. Prefer 11_build_campaign.py for executable commands."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
from absa_xfer.campaign import build_plan, CORE_CONDITIONS

SEEDS=[13,37,73]
DOMAINS=["restaurant","laptop"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="FacebookAI/roberta-base"); ap.add_argument("--out",required=True); a=ap.parse_args()
    rows=build_plan(DOMAINS,[a.model],CORE_CONDITIONS,SEEDS)
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} planned runs to {p}")
if __name__=="__main__": main()
