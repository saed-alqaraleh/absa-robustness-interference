#!/usr/bin/env python
import argparse,json
import pandas as pd
from absa_xfer.statistics import paired_cluster_bootstrap
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True);ap.add_argument("--b",required=True)
    ap.add_argument("--metric",choices=["accuracy","macro_f1"],default="accuracy")
    ap.add_argument("--n-boot",type=int,default=5000);ap.add_argument("--seed",type=int,default=13)
    a=ap.parse_args()
    out=paired_cluster_bootstrap(pd.read_csv(a.a),pd.read_csv(a.b),a.metric,a.n_boot,a.seed)
    print(json.dumps(out,indent=2))
if __name__=="__main__":
    main()
