import pandas as pd
from absa_xfer.statistics import paired_cluster_bootstrap

def test_bootstrap_runs():
    a=pd.DataFrame({
      "example_id":["1","2","3","4"],"cluster_id":["a","a","b","c"],
      "y_true":[0,0,1,2],"y_pred":[0,0,1,2]
    })
    b=a.copy(); b["y_pred"]=[0,1,1,1]
    out=paired_cluster_bootstrap(a,b,"accuracy",100,13)
    assert out["effect_a_minus_b"]>0
    assert out["n_clusters"]==3
