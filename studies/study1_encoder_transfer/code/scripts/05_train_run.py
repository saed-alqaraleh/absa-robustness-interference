#!/usr/bin/env python
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from absa_xfer.io_utils import read_jsonl,atomic_write_json
from absa_xfer.training import train_one_run
from absa_xfer.run_signature import compute_run_signature
from absa_xfer.hashing import hash_object, sha256_file
from absa_xfer.provenance import git_code_version, append_jsonl_locked, collect_environment


def _hash_path_or_fallback(path: str|None, fallback_obj) -> str:
    if path and Path(path).exists():
        return sha256_file(path)
    return hash_object(fallback_obj)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--dev",required=True)
    ap.add_argument("--eval",action="append",required=True,help="NAME=PATH; repeat")
    ap.add_argument("--model",required=True)
    ap.add_argument("--model-revision",default=None)
    ap.add_argument("--tokenizer-revision",default=None)
    ap.add_argument("--condition",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--max-length",type=int,default=128)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--epochs",type=float,default=8)
    ap.add_argument("--pair-order",default="sentence_first")
    ap.add_argument("--split-manifest",default=None)
    ap.add_argument("--transformation-manifest",default=None)
    ap.add_argument("--source-selection",default=None)
    ap.add_argument("--master-manifest",default=None)
    ap.add_argument("--repo-root",default=str(Path(__file__).resolve().parents[1]))
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    train=read_jsonl(args.train); dev=read_jsonl(args.dev)
    tests={}; eval_paths={}
    for item in args.eval:
        name,path=item.split("=",1); tests[name]=read_jsonl(path); eval_paths[name]=path

    hp={"max_length":args.max_length,"learning_rate":args.lr,"train_batch_size":args.batch,
        "eval_batch_size":max(args.batch,32),"epochs":args.epochs,"weight_decay":0.01,
        "warmup_ratio":0.1,"early_stopping_patience":2,"fp16":True,"bf16":False}

    # Resolve floating Hub names to an immutable commit before the run signature is computed.
    resolved_revision=args.model_revision
    if not resolved_revision:
        try:
            from huggingface_hub import HfApi
            resolved_revision=HfApi().model_info(args.model).sha
            print(f"Resolved {args.model} -> {resolved_revision}")
        except Exception as e:
            raise RuntimeError("Could not resolve an immutable Hugging Face model revision; use --model-revision") from e
    resolved_tokenizer_revision=args.tokenizer_revision or resolved_revision

    data_files={"train":sha256_file(args.train),"dev":sha256_file(args.dev),
                **{f"eval:{k}":sha256_file(v) for k,v in eval_paths.items()}}
    split_fallback={"train_sources":sorted({r["source_instance_id"] for r in train}),
                    "dev_sources":sorted({r["source_instance_id"] for r in dev})}
    transform_fallback=sorted({
        (r.get("perturbation_family"), (r.get("generation") or {}).get("generation_rule"),
         (r.get("generation") or {}).get("provenance"))
        for r in train if r.get("is_augmented")
    })
    source_fallback=sorted({r["source_instance_id"] for r in train if r.get("is_augmented")})
    code_version=git_code_version(args.repo_root)
    sig_meta={
        "dataset_hash":hash_object(data_files),
        "split_hash":_hash_path_or_fallback(args.split_manifest,split_fallback),
        "transformation_hash":_hash_path_or_fallback(args.transformation_manifest,transform_fallback),
        "source_selection_hash":_hash_path_or_fallback(args.source_selection,source_fallback),
        "model_name":args.model,
        "model_revision":resolved_revision,
        "tokenizer_name":args.model,
        "tokenizer_revision":resolved_tokenizer_revision,
        "seed":args.seed,
        "hyperparameters":hp,
        "code_version":code_version,
    }
    signature=compute_run_signature(sig_meta)
    meta={
        "run_id":out.name,"dataset":args.dataset,"model":args.model,
        "training_condition":args.condition,"seed":args.seed,
        "perturbation_budget":sum(bool(r.get("is_augmented")) for r in train),
        "run_signature":signature,"signature_meta":sig_meta,"input_file_hashes":data_files,
        "environment":collect_environment(),"status":"STARTED","start_time_unix":time.time(),
    }
    done_path=out/"DONE.json"
    if done_path.exists():
        done=json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("run_signature")==signature and done.get("status")=="complete":
            print("SKIP valid completed run:",out)
            return
        raise RuntimeError(f"Refusing incompatible completed output at {out}: signature mismatch")
    atomic_write_json(out/"run_meta.json",meta)

    state=train_one_run(
        train,dev,tests,args.model,out,args.seed,hp,
        model_revision=resolved_revision,tokenizer_revision=resolved_tokenizer_revision,
        pair_order=args.pair_order,run_signature=signature
    )
    meta["status"]="COMPLETE"; meta["end_time_unix"]=time.time(); meta["eval_summary"]=state["eval_summary"]
    atomic_write_json(out/"run_meta.json",meta)
    if args.master_manifest:
        append_jsonl_locked(args.master_manifest,meta)
    print(json.dumps(state["eval_summary"],indent=2))

if __name__=="__main__":
    main()
