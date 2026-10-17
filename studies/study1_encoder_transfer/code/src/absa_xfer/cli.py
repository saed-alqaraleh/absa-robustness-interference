from __future__ import annotations
import argparse, json
from pathlib import Path
from .bootstrap import main as bootstrap_main

def main():
    ap=argparse.ArgumentParser(prog="absa-xfer")
    sub=ap.add_subparsers(dest="cmd",required=True)

    p=sub.add_parser("bootstrap-sources")
    p.add_argument("--project-root",required=True)
    p.add_argument("--allow-update",action="store_true")

    p=sub.add_parser("hash-file")
    p.add_argument("path")

    args,unknown=ap.parse_known_args()
    if args.cmd=="bootstrap-sources":
        # lightweight delegation without relying on sys.argv mutation internals
        from .bootstrap import clone_or_update, catalog, REPOS
        from .io_utils import ensure_dir, atomic_write_json
        import sys,platform
        root=Path(args.project_root)
        raw=ensure_dir(root/"data_raw"/"external_repos")
        repos={}
        for name,url in REPOS.items():
            dest=raw/name
            commit=clone_or_update(url,dest,args.allow_update)
            repos[name]={"url":url,"commit":commit,"catalog":catalog(dest)}
        manifest={"python":sys.version,"platform":platform.platform(),"repos":repos}
        atomic_write_json(root/"manifests"/"source_repository_manifest.json",manifest)
        print(json.dumps({k:v["commit"] for k,v in repos.items()},indent=2))
    elif args.cmd=="hash-file":
        from .hashing import sha256_file
        print(sha256_file(args.path))

if __name__=="__main__":
    main()
