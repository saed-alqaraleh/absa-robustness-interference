#!/usr/bin/env python3
"""Offline, standard-library reconstruction of saved ABSA study results."""
import argparse,csv,json,hashlib,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABELS=('negative','neutral','positive')

def readrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def accuracy(rows):
    return sum(r['gold']==r['pred'] for r in rows)/len(rows)

def macro_f1(rows):
    score=0.
    for label in LABELS:
        tp=sum(r['gold']==label and r['pred']==label for r in rows)
        fp=sum(r['gold']!=label and r['pred']==label for r in rows)
        fn=sum(r['gold']==label and r['pred']!=label for r in rows)
        den=2*tp+fp+fn
        score+=2*tp/den if den else 0.
    return score/3

def close(x,y):
    if abs(x-y)>1e-10:raise ValueError(f'expected {y}, reconstructed {x}')

def consistent(rows,study,family,record):
    if not rows:raise ValueError(f'empty cohort: {record}')
    for r in rows:
        if r['gold'] not in LABELS or r['pred'] not in LABELS:raise ValueError(f'bad label {record}')
        if not r.get('source_instance_id') or not r.get('cluster_id'):raise ValueError(f'missing identifiers {record}')

def study1():
    base=ROOT/'studies/study1_encoder_transfer'
    n_runs=n_preds=n_metrics=0
    for p in sorted(base.glob('*/runs/**/run_result.json')):
        d=json.loads(p.read_text());n_runs+=1
        for fam,filename in d['prediction_files'].items():
            rows=readrows(p.parent/filename);consistent(rows,1,fam,p)
            close(accuracy(rows),d['eval_summary'][fam]['accuracy'])
            close(macro_f1(rows),d['eval_summary'][fam]['macro_f1'])
            n_preds+=1;n_metrics+=2
    assert (n_runs,n_preds,n_metrics)==(174,696,1392),(n_runs,n_preds,n_metrics)
    return {'runs':n_runs,'prediction_files':n_preds,'recomputed_metrics':n_metrics}

def study2():
    base=ROOT/'studies/study2_budget_strategies/runs'
    kinds=Counter();n_preds=n_metrics=0
    for p in sorted(base.glob('**/run_result.json')):
        d=json.loads(p.read_text());kinds[p.relative_to(base).parts[0]]+=1
        for fam,dstat in d['eval_summary'].items():
            rows=readrows(p.parent/f'predictions_{fam}.csv');consistent(rows,2,fam,p)
            close(accuracy(rows),dstat['accuracy'])
            close(macro_f1(rows),dstat['macro_f1'])
            assert len(rows)==dstat['n']
            n_preds+=1;n_metrics+=2
    assert dict(kinds)=={'confirm':54,'regression':6,'tuning':62},kinds
    assert (n_preds,n_metrics)==(488,976),(n_preds,n_metrics)
    return {'runs':sum(kinds.values()),'run_types':dict(kinds),'prediction_files':n_preds,'recomputed_metrics':n_metrics}

def study3():
    base=ROOT/'studies/study3_in_context_learning'
    n=0;values=0;rows_total=0
    for p in sorted(base.glob('*/*__seed*.csv')):
        done=json.loads(p.with_suffix('.DONE.json').read_text())
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==done['csv_sha256'],p
        rows=readrows(p);rows_total+=len(rows)
        grp=defaultdict(list)
        for r in rows:grp[r['set']].append(r)
        for fam,recorded in done['accuracy'].items():
            consistent(grp[fam],3,fam,p)
            close(accuracy(grp[fam]),recorded)
            assert len(grp[fam])==done['n'][fam]
            values+=1
        n+=1
    assert (n,values)==(64,256),(n,values)
    return {'jobs':n,'accuracy_checks':values,'prediction_rows':rows_total}

def study4():
    base=ROOT/'studies/study4_lora_adaptation/runs'
    runs=trained=zs=cohorts=jsichecks=0
    for p in sorted(base.glob('*/run_result.json')):
        d=json.loads(p.read_text());rows=readrows(p.parent/'predictions.csv')
        grp=defaultdict(list)
        for r in rows:grp[r['set']].append(r)
        acc={}
        for fam,recorded in d['accuracy'].items():
            consistent(grp[fam],4,fam,p);acc[fam]=accuracy(grp[fam]);close(acc[fam],recorded)
            assert len(grp[fam])==d['n'][fam]
            cohorts+=1
        close(sum(acc[k] for k in ('RevTgt','RevNon','AddDiff'))/3,d['R_mean'])
        close(min(acc[k] for k in ('RevTgt','RevNon','AddDiff')),d['W'])
        rt={r['source_instance_id']:r for r in grp['RevTgt']}
        rn={r['source_instance_id']:r for r in grp['RevNon']}
        shared=rt.keys()&rn.keys()
        assert len(shared)==d['n_J']
        j=sum(rt[k]['gold']==rt[k]['pred'] and rn[k]['gold']==rn[k]['pred'] for k in shared)/len(shared)
        close(j,d['J_SI']);jsichecks+=1;runs+=1
        if '__ZS__' in p.parent.name:zs+=1
        else:trained+=1
    assert (runs,trained,zs,cohorts,jsichecks)==(26,24,2,104,26)
    return {'jobs':runs,'fine_tunes':trained,'zero_shot':zs,'cohort_accuracy_checks':cohorts,'joint_satisfaction_checks':jsichecks}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['all','1','2','3','4'],default='all');a=ap.parse_args()
    funcs={'1':study1,'2':study2,'3':study3,'4':study4}
    output={k:fn() for k,fn in funcs.items() if a.study in ('all',k)}
    print(json.dumps({'status':'PASS','studies':output},indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FAIL:',type(exc).__name__,str(exc),file=sys.stderr);sys.exit(1)
