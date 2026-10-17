#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from absa_xfer.io_utils import atomic_write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/base.yaml'); ap.add_argument('--out',required=True); a=ap.parse_args()
    from huggingface_hub import HfApi
    cfg=yaml.safe_load(Path(a.config).read_text())
    models=list(dict.fromkeys([cfg['models']['minimal_decisive'],*cfg['models']['final']]))
    api=HfApi(); pins={}
    for model in models:
        info=api.model_info(model)
        pins[model]={'revision':info.sha,'last_modified':str(info.last_modified) if info.last_modified else None}
        print(model, info.sha)
    obj={'source':'Hugging Face Hub model_info','pins':pins}
    atomic_write_json(a.out,obj)
    print('Wrote',a.out)
if __name__=='__main__': main()
