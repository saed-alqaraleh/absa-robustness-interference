from __future__ import annotations
import hashlib, random
from collections import defaultdict
from .schemas import AspectExample
from .transformations import (
    generate_revtgt, generate_revnon, generate_adddiff,
    build_aspect_expression_library
)

def _stable_repeat_indices(n: int, budget: int, seed: int) -> list[int]:
    if n <= 0: return []
    rng = random.Random(seed)
    base = list(range(n))
    out = []
    while len(out) < budget:
        chunk = base[:]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out[:budget]

def materialize_conditions(
    train_examples: list[AspectExample],
    opinions,
    selected_ids: list[str],
    seed: int,
    mixed_total_budget: int | None = None
) -> dict[str, list[dict]]:
    by_cluster = defaultdict(list)
    by_id = {x.source_instance_id:x for x in train_examples}
    for x in train_examples:
        by_cluster[x.cluster_id].append(x)
    selected = [by_id[sid] for sid in selected_ids]
    library = build_aspect_expression_library(train_examples, opinions)
    B = mixed_total_budget if mixed_total_budget is not None else len(selected)
    clean_rows = [{
        "sentence":x.sentence, "aspect_term":x.aspect_term, "label":x.label,
        "source_instance_id":x.source_instance_id, "cluster_id":x.cluster_id,
        "is_augmented":False, "perturbation_family":"Clean"
    } for x in train_examples]
    conditions = {"Clean-Natural":clean_rows}

    # Clean matched repeats the selected controlled source pool.
    repeat_idx = _stable_repeat_indices(len(selected), B, seed)
    repeated = []
    for idx in repeat_idx:
        x = selected[idx]
        repeated.append({
            "sentence":x.sentence, "aspect_term":x.aspect_term, "label":x.label,
            "source_instance_id":x.source_instance_id, "cluster_id":x.cluster_id,
            "is_augmented":True, "perturbation_family":"CleanRepeat"
        })
    conditions["Clean-Matched"] = clean_rows + repeated

    generated_by_family = {}
    for family in ("RevTgt","RevNon","AddDiff"):
        g = []
        for x in selected:
            if family == "RevTgt":
                y = generate_revtgt(x, by_cluster[x.cluster_id], opinions, seed)
            elif family == "RevNon":
                y = generate_revnon(x, by_cluster[x.cluster_id], opinions, seed)
            else:
                y = generate_adddiff(x, by_cluster[x.cluster_id], opinions, library, seed)
            g.append(y)
        generated_by_family[family] = g
        # select/repeat to exact B deterministically
        idxs = _stable_repeat_indices(len(g), B, seed+{"RevTgt":1,"RevNon":2,"AddDiff":3}[family])
        rows = []
        for i in idxs:
            y = g[i]
            rows.append({
                "sentence":y.transformed_sentence, "aspect_term":y.transformed_aspect,
                "label":y.transformed_label, "source_instance_id":y.source_instance_id,
                "cluster_id":y.cluster_id, "is_augmented":True,
                "perturbation_family":family, "generation":y.to_dict()
            })
        conditions["Clean+"+family] = clean_rows + rows

    # Mixed gets B total, split as equally as possible.
    fams = ("RevTgt","RevNon","AddDiff")
    quotas = {f:B//3 for f in fams}
    for f in fams[:B%3]:
        quotas[f]+=1
    mixed = []
    for f in fams:
        g = generated_by_family[f]
        # deterministic family-specific order
        g = sorted(g, key=lambda y: hashlib.sha256(f"{seed}|Mixed|{f}|{y.source_instance_id}".encode()).hexdigest())
        idxs = _stable_repeat_indices(len(g), quotas[f], seed+11+fams.index(f))
        for i in idxs:
            y = g[i]
            mixed.append({
                "sentence":y.transformed_sentence, "aspect_term":y.transformed_aspect,
                "label":y.transformed_label, "source_instance_id":y.source_instance_id,
                "cluster_id":y.cluster_id, "is_augmented":True,
                "perturbation_family":f, "generation":y.to_dict()
            })
    assert len(mixed)==B
    conditions["Clean+Mixed"] = clean_rows + mixed
    return conditions


def materialize_lopo_conditions(
    train_examples: list[AspectExample],
    opinions,
    selected_ids: list[str],
    seed: int,
    total_budget: int | None = None,
) -> dict[str, list[dict]]:
    """Materialize the three fixed-total-budget leave-one-family-out conditions.

    The frozen common source pool is reused exactly.  Each LOPO condition contains
    exactly one derivative for every selected source instance, so the total
    augmentation budget is B (not 2B) and source-instance composition is identical
    across the three pairwise-family conditions.

    For odd B, the single extra example is rotated symmetrically across the three
    perturbation families: RevTgt in RevTgt+RevNon, AddDiff in RevTgt+AddDiff,
    and RevNon in RevNon+AddDiff.
    """
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("LOPO selected source IDs contain duplicates")

    by_cluster = defaultdict(list)
    by_id = {x.source_instance_id: x for x in train_examples}
    for x in train_examples:
        by_cluster[x.cluster_id].append(x)

    missing = [sid for sid in selected_ids if sid not in by_id]
    if missing:
        raise ValueError(f"LOPO selected source IDs missing from train set: {missing[:10]}")

    selected = [by_id[sid] for sid in selected_ids]
    B = total_budget if total_budget is not None else len(selected)
    if B != len(selected):
        raise ValueError(
            f"LOPO requires total_budget to equal the frozen selected-source count: "
            f"budget={B}, selected={len(selected)}"
        )
    if B <= 0:
        raise ValueError("LOPO budget must be positive")

    library = build_aspect_expression_library(train_examples, opinions)
    clean_rows = [{
        "sentence": x.sentence,
        "aspect_term": x.aspect_term,
        "label": x.label,
        "source_instance_id": x.source_instance_id,
        "cluster_id": x.cluster_id,
        "is_augmented": False,
        "perturbation_family": "Clean",
    } for x in train_examples]

    # Generate every family for every frozen selected source first.  The source
    # selection was pre-audited as generation-valid; any generation failure here
    # is therefore an integrity failure and is allowed to propagate fail-closed.
    generated: dict[str, dict[str, object]] = {}
    for family in ("RevTgt", "RevNon", "AddDiff"):
        vals: dict[str, object] = {}
        for x in selected:
            if family == "RevTgt":
                y = generate_revtgt(x, by_cluster[x.cluster_id], opinions, seed)
            elif family == "RevNon":
                y = generate_revnon(x, by_cluster[x.cluster_id], opinions, seed)
            else:
                y = generate_adddiff(x, by_cluster[x.cluster_id], opinions, library, seed)
            if y.source_instance_id != x.source_instance_id:
                raise AssertionError(
                    f"LOPO generation changed source lineage for {family}: "
                    f"{x.source_instance_id} -> {y.source_instance_id}"
                )
            vals[x.source_instance_id] = y
        if set(vals) != set(selected_ids):
            raise AssertionError(f"LOPO generation coverage mismatch for {family}")
        generated[family] = vals

    # (family A, family B, family receiving the +1 slot when B is odd)
    pair_specs = (
        ("RevTgt", "RevNon", "RevTgt"),
        ("RevTgt", "AddDiff", "AddDiff"),
        ("RevNon", "AddDiff", "RevNon"),
    )

    out: dict[str, list[dict]] = {}
    for fa, fb, odd_extra_family in pair_specs:
        condition = f"Clean+{fa}+{fb}"
        order = sorted(
            selected_ids,
            key=lambda sid: hashlib.sha256(
                f"{seed}|LOPO_SOURCE_PARTITION|{condition}|{sid}".encode()
            ).hexdigest(),
        )

        half = B // 2
        quotas = {fa: half, fb: half}
        if B % 2:
            quotas[odd_extra_family] += 1

        first_family = odd_extra_family if B % 2 else fa
        second_family = fb if first_family == fa else fa
        first_n = quotas[first_family]
        family_for_source = {
            sid: (first_family if idx < first_n else second_family)
            for idx, sid in enumerate(order)
        }

        aug: list[dict] = []
        for sid in order:
            fam = family_for_source[sid]
            y = generated[fam][sid]
            aug.append({
                "sentence": y.transformed_sentence,
                "aspect_term": y.transformed_aspect,
                "label": y.transformed_label,
                "source_instance_id": y.source_instance_id,
                "cluster_id": y.cluster_id,
                "is_augmented": True,
                "perturbation_family": fam,
                "generation": y.to_dict(),
            })

        if len(aug) != B:
            raise AssertionError(f"LOPO budget mismatch for {condition}: {len(aug)} != {B}")
        aug_ids = [r["source_instance_id"] for r in aug]
        if len(set(aug_ids)) != B or set(aug_ids) != set(selected_ids):
            raise AssertionError(f"LOPO source-composition mismatch for {condition}")
        fam_counts = {f: sum(r["perturbation_family"] == f for r in aug) for f in (fa, fb)}
        if fam_counts != quotas:
            raise AssertionError(
                f"LOPO family quota mismatch for {condition}: {fam_counts} != {quotas}"
            )

        out[condition] = clean_rows + aug

    return out


def materialize_lopo_from_frozen_rows(
    clean_rows: list[dict],
    family_rows: dict[str, list[dict]],
    selected_ids: list[str],
    seed: int,
) -> dict[str, list[dict]]:
    """Compose LOPO conditions from already-frozen single-family derivatives.

    No transformation is regenerated.  Each family must supply exactly one
    augmented row for every frozen selected source instance.  The pairwise LOPO
    conditions then choose one frozen family row per source at total budget B.
    """
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("LOPO selected source IDs contain duplicates")
    if not selected_ids:
        raise ValueError("LOPO selected source pool is empty")

    B = len(selected_ids)
    selected_set = set(selected_ids)
    expected_families = {"RevTgt", "RevNon", "AddDiff"}
    if set(family_rows) != expected_families:
        raise ValueError(
            f"LOPO frozen family rows must be exactly {sorted(expected_families)}; "
            f"found {sorted(family_rows)}"
        )

    by_family_source: dict[str, dict[str, dict]] = {}
    for family in ("RevTgt", "RevNon", "AddDiff"):
        rows = family_rows[family]
        mapping: dict[str, dict] = {}
        for row in rows:
            if not bool(row.get("is_augmented")):
                raise ValueError(f"Frozen {family} row is not marked augmented")
            if row.get("perturbation_family") != family:
                raise ValueError(
                    f"Frozen {family} input contains perturbation_family={row.get('perturbation_family')!r}"
                )
            sid = str(row.get("source_instance_id"))
            if sid in mapping:
                raise ValueError(f"Frozen {family} input duplicates source {sid}")
            mapping[sid] = row
        if set(mapping) != selected_set:
            missing = sorted(selected_set - set(mapping))
            extra = sorted(set(mapping) - selected_set)
            raise ValueError(
                f"Frozen {family} source coverage mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        by_family_source[family] = mapping

    pair_specs = (
        ("RevTgt", "RevNon", "RevTgt"),
        ("RevTgt", "AddDiff", "AddDiff"),
        ("RevNon", "AddDiff", "RevNon"),
    )
    out: dict[str, list[dict]] = {}
    for fa, fb, odd_extra_family in pair_specs:
        condition = f"Clean+{fa}+{fb}"
        order = sorted(
            selected_ids,
            key=lambda sid: hashlib.sha256(
                f"{seed}|LOPO_SOURCE_PARTITION|{condition}|{sid}".encode()
            ).hexdigest(),
        )
        half = B // 2
        quotas = {fa: half, fb: half}
        if B % 2:
            quotas[odd_extra_family] += 1

        first_family = odd_extra_family if B % 2 else fa
        second_family = fb if first_family == fa else fa
        first_n = quotas[first_family]

        aug: list[dict] = []
        for idx, sid in enumerate(order):
            family = first_family if idx < first_n else second_family
            # Copy the dict so downstream writing cannot mutate the frozen input.
            aug.append(dict(by_family_source[family][sid]))

        if len(aug) != B:
            raise AssertionError(f"LOPO budget mismatch for {condition}: {len(aug)} != {B}")
        if {r["source_instance_id"] for r in aug} != selected_set:
            raise AssertionError(f"LOPO source-composition mismatch for {condition}")
        counts = {
            fam: sum(r["perturbation_family"] == fam for r in aug)
            for fam in (fa, fb)
        }
        if counts != quotas:
            raise AssertionError(f"LOPO quota mismatch for {condition}: {counts} != {quotas}")
        out[condition] = [dict(row) for row in clean_rows] + aug

    return out
