#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from absa_xfer.semeval import parse_semeval_xml
from absa_xfer.splits import make_group_stratified_dev
from absa_xfer.io_utils import write_jsonl, atomic_write_json
from absa_xfer.hashing import sha256_file, hash_object

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--domain",choices=["restaurant","laptop"],required=True)
    ap.add_argument("--train-xml",required=True)
    ap.add_argument("--test-xml",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--dev-size",type=int,default=150)
    ap.add_argument("--split-seed",type=int,default=13)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    raw_train=parse_semeval_xml(args.train_xml,args.domain,"raw_train")
    test=parse_semeval_xml(args.test_xml,args.domain,"test")
    train,dev=make_group_stratified_dev(raw_train,args.dev_size,args.split_seed)
    # relabel split names immutably by writing dictionaries
    def rows(xs, split):
        return [{**x.to_dict(),"split":split} for x in xs]
    tr,dv,te=rows(train,"train"),rows(dev,"dev"),rows(test,"test")
    write_jsonl(out/"train.jsonl",tr);write_jsonl(out/"dev.jsonl",dv);write_jsonl(out/"test.jsonl",te)
    manifest={
      "domain":args.domain,
      "source_files":{
        "train":{"path":args.train_xml,"sha256":sha256_file(args.train_xml)},
        "test":{"path":args.test_xml,"sha256":sha256_file(args.test_xml)}
      },
      "split_policy":"deterministic sentence-level stratified fallback; replace with exact Xu IDs if recovered",
      "split_seed":args.split_seed,
      "counts":{"train":len(tr),"dev":len(dv),"test":len(te)},
      "dev_source_ids":[r["source_instance_id"] for r in dv],
    }
    manifest["manifest_hash"]=hash_object(manifest)
    atomic_write_json(out/"split_manifest.json",manifest)
    print(json.dumps(manifest["counts"],indent=2))

if __name__=="__main__":
    main()
