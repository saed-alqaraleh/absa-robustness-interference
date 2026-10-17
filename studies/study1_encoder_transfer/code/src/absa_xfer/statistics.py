from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def _metric(y, p, metric: str):
    if metric=="accuracy":
        return accuracy_score(y,p)
    if metric=="macro_f1":
        return f1_score(y,p,average="macro",labels=[0,1,2],zero_division=0)
    raise ValueError(metric)

def paired_cluster_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str = "accuracy",
    n_boot: int = 5000,
    seed: int = 13,
    cluster_col: str = "cluster_id"
) -> dict:
    """
    Compare A-B on the same evaluation rows. Dataframes need columns:
    example_id, cluster_id, y_true, y_pred.
    Resampling unit is original source sentence cluster.
    """
    need = {"example_id",cluster_col,"y_true","y_pred"}
    if not need.issubset(a.columns) or not need.issubset(b.columns):
        raise ValueError(f"missing required columns: {need}")
    aa = a[["example_id",cluster_col,"y_true","y_pred"]].rename(columns={"y_pred":"pred_a"})
    bb = b[["example_id","y_pred"]].rename(columns={"y_pred":"pred_b"})
    m = aa.merge(bb,on="example_id",how="inner",validate="one_to_one")
    if len(m)!=len(a) or len(m)!=len(b):
        raise ValueError("prediction sets differ; paired comparison invalid")
    point = _metric(m.y_true,m.pred_a,metric)-_metric(m.y_true,m.pred_b,metric)
    clusters = m[cluster_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot,dtype=float)
    grouped = {c:m[m[cluster_col]==c] for c in clusters}
    for i in range(n_boot):
        sampled = rng.choice(clusters,size=len(clusters),replace=True)
        boot = pd.concat([grouped[c] for c in sampled],ignore_index=True)
        vals[i] = _metric(boot.y_true,boot.pred_a,metric)-_metric(boot.y_true,boot.pred_b,metric)
    lo,hi = np.quantile(vals,[0.025,0.975])
    return {
        "metric":metric,"effect_a_minus_b":float(point),
        "ci95_low":float(lo),"ci95_high":float(hi),
        "n_clusters":int(len(clusters)),"n_rows":int(len(m)),
        "n_boot":int(n_boot),"bootstrap_seed":int(seed)
    }
