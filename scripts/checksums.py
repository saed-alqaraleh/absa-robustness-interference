#!/usr/bin/env python3
"""Verify or generate a SHA-256 inventory of repository contents."""
import argparse,hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=root/'reproducibility/SHA256SUMS.txt'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():
    return sorted(p for p in root.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts and '.git' not in p.parts)

def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify',action='store_true');g.add_argument('--generate',action='store_true');args=parser.parse_args()
    expected={p.relative_to(root).as_posix():digest(p) for p in files()}
    if args.generate:
        manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_text(''.join(f'{h}  {n}\n' for n,h in sorted(expected.items())))
        print('Generated checksum inventory for',len(expected),'files');return
    recorded={}
    for line in manifest.read_text().splitlines():
        h,n=line.split('  ',1)
        if n in recorded:raise ValueError('duplicate manifest entry: '+n)
        recorded[n]=h
    missing=sorted(recorded.keys()-expected.keys());extra=sorted(expected.keys()-recorded.keys())
    invalid=sorted(k for k in recorded.keys()&expected.keys() if recorded[k]!=expected[k])
    if missing or extra or invalid:raise ValueError(f'checksums failed: missing={missing[:5]} extra={extra[:5]} invalid={invalid[:5]}')
    print('PASS:',len(recorded),'file SHA-256 checksums')
if __name__=='__main__':
    try:main()
    except Exception as e:print('FAIL:',e,file=sys.stderr);sys.exit(1)
