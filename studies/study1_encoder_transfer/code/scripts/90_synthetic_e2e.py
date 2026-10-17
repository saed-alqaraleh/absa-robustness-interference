#!/usr/bin/env python
"""Non-neural synthetic end-to-end preflight.

Exercises source clustering, eligibility, common-source selection, all six core
conditions, LOPO fixed-budget materialization, leakage audit, and paired cluster
bootstrap without downloading datasets or model weights.
"""
from __future__ import annotations
import json,tempfile
from pathlib import Path
from absa_xfer.schemas import AspectExample, OpinionRecord, OpinionSpan
from absa_xfer.transformations import build_aspect_expression_library
from absa_xfer.source_control import audit_eligibility,select_common_pool
from absa_xfer.materialize import materialize_conditions,materialize_lopo_conditions
from absa_xfer.integrity import integrity_report
from absa_xfer.statistics import paired_cluster_bootstrap
import absa_xfer.transformations as t
import absa_xfer.materialize as m


def ex(domain,split,sid,sentence,term,start,end,label,aid):
    return AspectExample(domain,split,sid,sentence,term,start,end,label,aid,sid)

def main():
    train=[]; ops={}
    fixtures=[
      ("s1","Tasty burgers and crispy fries.",[("burgers",6,13,"positive","Tasty",0,5),("fries",25,30,"positive","crispy",18,24)]),
      ("s2","Awful pasta and rude service.",[("pasta",6,11,"negative","Awful",0,5),("service",21,28,"negative","rude",16,20)]),
      ("s3","Great soup but bad music.",[("soup",6,10,"positive","Great",0,5),("music",19,24,"negative","bad",15,18)]),
      ("s4","Poor drinks but kind staff.",[("drinks",5,11,"negative","Poor",0,4),("staff",21,26,"positive","kind",16,20)]),
    ]
    for sid,sent,aspects in fixtures:
        for j,(term,s,e,label,op,os,oe) in enumerate(aspects):
            aid=f"{sid}a{j}"; x=ex("restaurant","train",sid,sent,term,s,e,label,aid); train.append(x)
            ops[aid]=OpinionRecord(aid,sid,(OpinionSpan(os,oe,op,"synthetic"),),"synthetic","synthetic")
    # deterministic explicit lexicon; no external corpus needed.
    lex={"tasty":["awful"],"crispy":["soggy"],"awful":["tasty"],"rude":["kind"],"great":["poor"],"bad":["good"],"poor":["great"],"kind":["rude"]}
    orig=t.reverse_opinion_text
    def patched(span,key,antonym_lexicon=None): return orig(span,key,lex)
    t.reverse_opinion_text=patched
    m.generate_revtgt=lambda x,asps,op,seed: t.generate_revtgt(x,asps,op,seed,lex)
    m.generate_revnon=lambda x,asps,op,seed: t.generate_revnon(x,asps,op,seed,lex)

    lib=build_aspect_expression_library(train,ops)
    erows,counts=audit_eligibility(train,ops,lib)
    selected=select_common_pool(erows,None,13)
    core=materialize_conditions(train,ops,selected,13,mixed_total_budget=len(selected))
    lopo=materialize_lopo_conditions(train,ops,selected,13,total_budget=len(selected))
    # Disjoint tiny dev/test source fixtures for integrity path.
    dev=[ex("restaurant","dev","d1","Nice tea.","tea",5,8,"positive","d1a")]
    test=[ex("restaurant","test","t1","Bad coffee.","coffee",4,10,"negative","t1a")]
    rep=integrity_report([x.to_dict() for x in train],[x.to_dict() for x in dev],[x.to_dict() for x in test],core["Clean+Mixed"])
    assert not rep["blocking"]
    for name,rows in core.items():
        if name!="Clean-Natural": assert sum(r["is_augmented"] for r in rows)==len(selected)
    for rows in lopo.values(): assert sum(r["is_augmented"] for r in rows)==len(selected)
    import pandas as pd
    pred_a=pd.DataFrame([{"example_id":"e1","cluster_id":"a","y_true":1,"y_pred":1},{"example_id":"e2","cluster_id":"b","y_true":0,"y_pred":1}])
    pred_b=pd.DataFrame([{"example_id":"e1","cluster_id":"a","y_true":1,"y_pred":1},{"example_id":"e2","cluster_id":"b","y_true":0,"y_pred":0}])
    stat=paired_cluster_bootstrap(pred_a,pred_b,"accuracy",n_boot=200,seed=13)
    summary={"status":"PASS","train_instances":len(train),"aspect_library":len(lib),"eligibility":counts,
             "selected_common_pool":len(selected),"core_conditions":{k:len(v) for k,v in core.items()},
             "lopo_conditions":{k:len(v) for k,v in lopo.items()},"integrity_blocking":rep["blocking"],"bootstrap_smoke":stat}
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
