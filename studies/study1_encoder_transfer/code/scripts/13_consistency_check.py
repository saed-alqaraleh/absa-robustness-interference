#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from absa_xfer.consistency import campaign_consistency
from absa_xfer.io_utils import atomic_write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plan",required=True); ap.add_argument("--results",required=True); ap.add_argument("--runs",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); rep=campaign_consistency(a.plan,a.results,a.runs); atomic_write_json(a.out,rep); print(json.dumps(rep,indent=2));
    if not rep["complete"]: sys.exit(4)
if __name__=="__main__": main()
