from __future__ import annotations
import yaml
from pathlib import Path

def load_config(path: str | Path) -> dict:
    with open(path,encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    if not isinstance(cfg,dict):
        raise ValueError("config must be a mapping")
    return cfg
