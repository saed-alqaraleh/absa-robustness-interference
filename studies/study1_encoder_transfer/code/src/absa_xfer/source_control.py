from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict
from .schemas import AspectExample
from .transformations import eligibility, AspectExpression

FAMILIES = ("RevTgt","RevNon","AddDiff")

def sentence_length_bin(n_tokens: int) -> str:
    if n_tokens < 10: return "<10"
    if n_tokens < 20: return "10-19"
    if n_tokens < 30: return "20-29"
    return "30+"

def aspect_count_bin(n: int) -> str:
    if n <= 1: return "1"
    if n == 2: return "2"
    return "3+"

def audit_eligibility(
    examples: list[AspectExample],
    opinions,
    aspect_library: list[AspectExpression]
) -> tuple[list[dict], dict]:
    by_cluster = defaultdict(list)
    for x in examples:
        by_cluster[x.cluster_id].append(x)
    rows = []
    counts = Counter()
    for x in examples:
        fam = {}
        reasons = {}
        for f in FAMILIES:
            ok, reason = eligibility(x, by_cluster[x.cluster_id], opinions, f, aspect_library)
            fam[f] = bool(ok); reasons[f] = reason
            if ok: counts[f] += 1
        all3 = all(fam.values())
        if all3: counts["all3"] += 1
        rows.append({
            **x.to_dict(),
            **{f"eligible_{f}":fam[f] for f in FAMILIES},
            **{f"reason_{f}":reasons[f] for f in FAMILIES},
            "eligible_all3":all3,
            "aspect_count":len(by_cluster[x.cluster_id]),
            "aspect_count_bin":aspect_count_bin(len(by_cluster[x.cluster_id])),
            "sentence_length":len(x.sentence.split()),
            "sentence_length_bin":sentence_length_bin(len(x.sentence.split()))
        })
    # intersections
    for mask in range(1, 8):
        key = "&".join(FAMILIES[i] for i in range(3) if mask & (1<<i))
        counts["intersection_exact_"+key] = sum(
            all(r[f"eligible_{FAMILIES[i]}"] == bool(mask&(1<<i)) for i in range(3))
            for r in rows
        )
    return rows, dict(counts)

def select_common_pool(
    eligibility_rows: list[dict],
    budget: int | None,
    seed: int
) -> list[str]:
    candidates = [r for r in eligibility_rows if r["eligible_all3"]]
    # deterministic hash ranking within predeclared strata
    by_stratum = defaultdict(list)
    for r in candidates:
        stratum = (r["domain"], r["label"], r["aspect_count_bin"], r["sentence_length_bin"])
        by_stratum[stratum].append(r)
    for stratum, vals in by_stratum.items():
        vals.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['source_instance_id']}".encode()).hexdigest())
    if budget is None or budget >= len(candidates):
        return sorted(r["source_instance_id"] for r in candidates)
    # proportional largest-remainder allocation
    total = len(candidates)
    quotas = {s: budget * len(v) / total for s,v in by_stratum.items()}
    alloc = {s:int(q) for s,q in quotas.items()}
    remaining = budget - sum(alloc.values())
    order = sorted(quotas, key=lambda s: (-(quotas[s]-alloc[s]), str(s)))
    for s in order[:remaining]:
        alloc[s] += 1
    selected = []
    for s, vals in by_stratum.items():
        selected.extend(r["source_instance_id"] for r in vals[:alloc[s]])
    assert len(selected) == budget
    return sorted(selected)

def audit_selected_pool(rows: list[dict], selected_ids: set[str]) -> dict:
    sub = [r for r in rows if r["source_instance_id"] in selected_ids]
    return {
        "n":len(sub),
        "label_distribution":dict(Counter(r["label"] for r in sub)),
        "aspect_count_bin":dict(Counter(r["aspect_count_bin"] for r in sub)),
        "sentence_length_bin":dict(Counter(r["sentence_length_bin"] for r in sub)),
        "domain":dict(Counter(r["domain"] for r in sub)),
    }
