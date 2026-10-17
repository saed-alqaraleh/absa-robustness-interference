
from __future__ import annotations

from collections import Counter

from .firewall import normalize_sentence


def integrity_report(
    train_rows,
    dev_rows,
    test_rows,
    derivatives=None,
) -> dict:

    derivatives = derivatives or []

    issues = []

    # ----------------------------------------------------------------
    # Source-instance overlap
    # ----------------------------------------------------------------

    def ids(rows):
        return {
            r["source_instance_id"]
            for r in rows
        }

    tr = ids(train_rows)
    dv = ids(dev_rows)
    te = ids(test_rows)

    if tr & dv:
        issues.append({
            "severity": "BLOCK",
            "type": "train_dev_source_overlap",
            "n": len(tr & dv),
        })

    if tr & te:
        issues.append({
            "severity": "BLOCK",
            "type": "train_test_source_overlap",
            "n": len(tr & te),
        })

    if dv & te:
        issues.append({
            "severity": "BLOCK",
            "type": "dev_test_source_overlap",
            "n": len(dv & te),
        })

    # ----------------------------------------------------------------
    # Sentence-cluster overlap
    # ----------------------------------------------------------------

    tr_cluster = {
        r["cluster_id"]
        for r in train_rows
    }

    dv_cluster = {
        r["cluster_id"]
        for r in dev_rows
    }

    te_cluster = {
        r["cluster_id"]
        for r in test_rows
    }

    for a, b, name in [
        (
            tr_cluster,
            dv_cluster,
            "train_dev",
        ),
        (
            tr_cluster,
            te_cluster,
            "train_test",
        ),
        (
            dv_cluster,
            te_cluster,
            "dev_test",
        ),
    ]:

        overlap = a & b

        if overlap:
            issues.append({
                "severity": "BLOCK",
                "type":
                    name + "_cluster_overlap",
                "n": len(overlap),
            })

    # ----------------------------------------------------------------
    # Protected-test ancestry
    # ----------------------------------------------------------------

    protected = te

    leaked_derivatives = [
        d
        for d in derivatives
        if d.get("source_instance_id")
        in protected
    ]

    if leaked_derivatives:
        issues.append({
            "severity": "BLOCK",
            "type":
                "test_derivative_in_training",
            "n": len(leaked_derivatives),
        })

    # ----------------------------------------------------------------
    # CONTENT-LEVEL CROSS-SPLIT FIREWALL
    # ----------------------------------------------------------------

    norm_sets = {
        "train": {
            normalize_sentence(
                r["sentence"]
            )
            for r in train_rows
        },

        "dev": {
            normalize_sentence(
                r["sentence"]
            )
            for r in dev_rows
        },

        "test": {
            normalize_sentence(
                r["sentence"]
            )
            for r in test_rows
        },
    }

    for a, b, name in [
        (
            "train",
            "dev",
            "train_dev",
        ),
        (
            "train",
            "test",
            "train_test",
        ),
        (
            "dev",
            "test",
            "dev_test",
        ),
    ]:

        overlap = (
            norm_sets[a]
            &
            norm_sets[b]
        )

        if overlap:
            issues.append({
                "severity": "BLOCK",
                "type":
                    name
                    + "_normalized_sentence_overlap",
                "n": len(overlap),
            })

    # ----------------------------------------------------------------
    # Duplicate inventory
    # ----------------------------------------------------------------

    all_norm = Counter(
        normalize_sentence(
            r["sentence"]
        )
        for r in (
            train_rows
            + dev_rows
            + test_rows
        )
    )

    return {
        "counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "derivatives":
                len(derivatives),
        },

        "label_distribution": {
            "train": dict(
                Counter(
                    r["label"]
                    for r in train_rows
                )
            ),

            "dev": dict(
                Counter(
                    r["label"]
                    for r in dev_rows
                )
            ),

            "test": dict(
                Counter(
                    r["label"]
                    for r in test_rows
                )
            ),
        },

        "duplicate_normalized_sentences_total":
            sum(
                v - 1
                for v in all_norm.values()
                if v > 1
            ),

        "issues":
            issues,

        "blocking":
            any(
                x["severity"] == "BLOCK"
                for x in issues
            ),
    }
