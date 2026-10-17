#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
from absa_xfer.arts_discovery import discover_arts_files
from absa_xfer.io_utils import atomic_write_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    report=discover_arts_files(a.repo)
    atomic_write_json(a.out,report)
    print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2))
    print(f"Cataloged {report['n_candidate_files']} candidate data files -> {a.out}")
    if not report["automatic_family_mapping_safe"]:
        print("GATE OPEN: freeze an explicit Clean/RevTgt/RevNon/AddDiff file mapping before ingestion.")

if __name__=="__main__": main()
