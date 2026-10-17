from __future__ import annotations
import random
from collections import Counter, defaultdict
from .schemas import AspectExample

def split_by_frozen_ids(examples: list[AspectExample], dev_source_ids: set[str]) -> tuple[list[AspectExample], list[AspectExample]]:
    train, dev = [], []
    for x in examples:
        (dev if x.source_instance_id in dev_source_ids else train).append(x)
    return train, dev

def make_group_stratified_dev(
    examples: list[AspectExample],
    target_aspect_instances: int = 150,
    seed: int = 13
) -> tuple[list[AspectExample], list[AspectExample]]:
    """
    Deterministic sentence-level split fallback.

    It greedily samples entire sentence clusters while tracking class proportions.
    Because clusters may contain multiple aspects, the resulting dev count may differ
    slightly from target_aspect_instances. This fallback must be reported as a deviation
    from the published Xu/ARTS split if exact IDs are unavailable.
    """
    by_cluster = defaultdict(list)
    for x in examples:
        by_cluster[x.cluster_id].append(x)

    total = Counter(x.label for x in examples)
    total_n = len(examples)
    target_prop = {k: v / total_n for k, v in total.items()}

    rng = random.Random(seed)
    clusters = list(by_cluster.items())
    rng.shuffle(clusters)
    clusters.sort(key=lambda kv: (len(kv[1]), kv[0]))

    chosen = []
    cur = Counter()
    cur_n = 0

    def score(add):
        n2 = cur_n + len(add)
        c2 = cur.copy()
        c2.update(x.label for x in add)
        size_pen = abs(n2 - target_aspect_instances) / max(1, target_aspect_instances)
        dist_pen = sum(abs((c2[k] / max(1, n2)) - target_prop.get(k, 0.0)) for k in target_prop)
        return size_pen + 0.5 * dist_pen

    remaining = clusters[:]
    while remaining and cur_n < target_aspect_instances:
        best_idx = min(range(len(remaining)), key=lambda i: score(remaining[i][1]))
        cid, add = remaining.pop(best_idx)
        chosen.append(cid)
        cur.update(x.label for x in add)
        cur_n += len(add)

    chosen = set(chosen)
    dev = [x for x in examples if x.cluster_id in chosen]
    train = [x for x in examples if x.cluster_id not in chosen]
    return train, dev
