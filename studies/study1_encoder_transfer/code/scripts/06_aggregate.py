#!/usr/bin/env python
import argparse
from absa_xfer.aggregate import aggregate_run_results
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-root",required=True)
    ap.add_argument("--out-csv",required=True)
    ap.add_argument("--out-parquet",required=True)
    a=ap.parse_args()
    df=aggregate_run_results(a.runs_root,a.out_csv,a.out_parquet)
    print(df.to_string(index=False))
if __name__=="__main__":
    main()
