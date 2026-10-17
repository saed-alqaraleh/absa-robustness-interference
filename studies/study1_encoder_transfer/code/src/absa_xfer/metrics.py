from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

LABEL2ID = {"negative":0,"neutral":1,"positive":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def classification_metrics(y_true, y_pred) -> dict[str,float]:
    return {
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "macro_f1":float(f1_score(y_true,y_pred,average="macro",labels=[0,1,2],zero_division=0))
    }

def ars_score(rows: list[dict]) -> float:
    """
    ARS requires one row per source+variation prediction with group_id identifying
    the original source unit. A group is correct iff every required member is correct.
    """
    groups = {}
    for r in rows:
        gid = r["group_id"]
        groups.setdefault(gid, []).append(bool(r["correct"]))
    if not groups:
        return float("nan")
    return sum(all(v) for v in groups.values())/len(groups)
