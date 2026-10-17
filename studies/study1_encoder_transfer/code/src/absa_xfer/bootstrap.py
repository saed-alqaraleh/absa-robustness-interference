from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from pathlib import Path
from .hashing import sha256_file
from .io_utils import ensure_dir, atomic_write_json

REPOS = {
    "ARTS_TestSet":"https://github.com/zhijing-jin/ARTS_TestSet.git",
    "TOWE":"https://github.com/NJUNLP/TOWE.git",
}

def run(cmd, cwd=None):
    print("+", " ".join(map(str,cmd)))
    subprocess.run(cmd,cwd=cwd,check=True)

def clone_or_update(url: str, dest: Path, allow_update: bool=False):
    if not dest.exists():
        run(["git","clone","--depth","1",url,str(dest)])
    elif allow_update:
        run(["git","fetch","--all"],cwd=dest)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=dest,text=True).strip()
    return commit

def catalog(root: Path):
    rows=[]
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            rows.append({"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",required=True)
    ap.add_argument("--allow-update",action="store_true")
    args=ap.parse_args()
    root=Path(args.project_root)
    raw=ensure_dir(root/"data_raw"/"external_repos")
    repos={}
    for name,url in REPOS.items():
        dest=raw/name
        commit=clone_or_update(url,dest,args.allow_update)
        repos[name]={"url":url,"commit":commit,"catalog":catalog(dest)}
    manifest={
        "python":sys.version,"platform":platform.platform(),
        "repos":repos
    }
    atomic_write_json(root/"manifests"/"source_repository_manifest.json",manifest)
    print(json.dumps({k:v["commit"] for k,v in repos.items()},indent=2))

if __name__=="__main__":
    main()
