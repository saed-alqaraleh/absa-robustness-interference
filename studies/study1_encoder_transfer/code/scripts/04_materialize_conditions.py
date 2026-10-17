#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
from absa_xfer.io_utils import read_jsonl,write_jsonl,atomic_write_json
from absa_xfer.schemas import AspectExample,OpinionRecord,OpinionSpan
from absa_xfer.materialize import materialize_conditions
from absa_xfer.hashing import hash_object,sha256_file


def load_examples(path):
    return [AspectExample(**{k:r[k] for k in AspectExample.__dataclass_fields__}) for r in read_jsonl(path)]

def load_opinions(path):
    out={}
    for r in read_jsonl(path):
        spans=tuple(OpinionSpan(**s) for s in r["spans"])
        rec=OpinionRecord(r["source_instance_id"],r["cluster_id"],spans,r["matched_by"],r["provenance"])
        out[rec.source_instance_id]=rec
    return out

def load_selected(path: str) -> list[str]:
    p=Path(path)
    if p.suffix.lower()==".json":
        obj=json.loads(p.read_text(encoding="utf-8"))
        return list(obj["selected_ids"] if isinstance(obj,dict) else obj)
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--opinions",required=True)
    ap.add_argument("--selected-ids",required=True)
    ap.add_argument("--out-dir",required=True,help="conditions root; one directory per condition")
    ap.add_argument("--manifest-out",default=None)
    ap.add_argument("--seed",type=int,default=20260913)
    ap.add_argument("--budget",type=int,default=None)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    ex=load_examples(args.train);op=load_opinions(args.opinions);ids=load_selected(args.selected_ids)
    B=args.budget if args.budget is not None else len(ids)
    conditions=materialize_conditions(ex,op,ids,args.seed,mixed_total_budget=B)
    manifest={
        "implementation":"ARTS-inspired deterministic v1",
        "seed":args.seed,"budget":B,
        "source_selection_sha256":sha256_file(args.selected_ids),
        "train_sha256":sha256_file(args.train),"opinions_sha256":sha256_file(args.opinions),
        "conditions":{}
    }
    for name,rows in conditions.items():
        p=out/name/"train.jsonl"; p.parent.mkdir(parents=True,exist_ok=True); write_jsonl(p,rows)
        aug=sum(bool(r.get("is_augmented")) for r in rows)
        manifest["conditions"][name]={"path":str(p),"sha256":sha256_file(p),"rows":len(rows),"aug_or_repeat_records":aug}
        if name!="Clean-Natural" and aug != B:
            raise AssertionError(f"budget mismatch for {name}: {aug} != {B}")
    manifest["manifest_hash"]=hash_object(manifest)
    domain=ex[0].domain if ex else "unknown"
    mout=Path(args.manifest_out) if args.manifest_out else out.parents[2]/"manifests"/domain/"transformation_manifest.json"
    atomic_write_json(mout,manifest)
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
