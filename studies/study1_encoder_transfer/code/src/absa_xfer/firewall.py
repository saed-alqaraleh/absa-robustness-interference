
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter


def normalize_sentence(s: str) -> str:
    """
    Canonical sentence normalization used only for duplicate/leakage detection.

    This does NOT alter model input text.
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = s.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def sentence_content_hash(s: str) -> str:
    return hashlib.sha256(
        normalize_sentence(s).encode("utf-8")
    ).hexdigest()


def _norms(rows):
    return {
        normalize_sentence(r["sentence"])
        for r in rows
    }


def apply_content_firewall(
    train_rows,
    dev_rows,
    test_rows,
):
    """
    Leakage priority:

        protected test > frozen dev > train

    The protected test set is NEVER modified.

    If dev overlaps with test by normalized sentence text,
    the function stops because repairing that case would require
    changing the frozen development split.

    Training rows whose sentence text appears in dev or test
    are excluded from the canonical training set.
    """

    dev_norm = _norms(dev_rows)
    test_norm = _norms(test_rows)

    dev_test_overlap = dev_norm & test_norm

    if dev_test_overlap:
        raise ValueError(
            "Blocking dev/test normalized-sentence overlap: "
            f"{len(dev_test_overlap)}"
        )

    safe_train = []
    excluded = []

    for row in train_rows:

        n = normalize_sentence(row["sentence"])

        reasons = []

        if n in test_norm:
            reasons.append(
                "duplicate_of_protected_test"
            )

        if n in dev_norm:
            reasons.append(
                "duplicate_of_frozen_dev"
            )

        if reasons:

            excluded.append({
                **row,
                "firewall_reasons": reasons,
                "normalized_sentence_sha256":
                    sentence_content_hash(
                        row["sentence"]
                    ),
            })

        else:
            safe_train.append(row)

    # Hard postconditions.
    train_norm = _norms(safe_train)

    assert not (train_norm & dev_norm)
    assert not (train_norm & test_norm)
    assert not (dev_norm & test_norm)

    report = {
        "normalization":
            "Unicode NFKC + casefold + whitespace collapse",

        "priority":
            "protected_test > frozen_dev > train",

        "before_counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
        },

        "after_counts": {
            "train": len(safe_train),
            "dev": len(dev_rows),
            "test": len(test_rows),
        },

        "excluded_train_rows":
            len(excluded),

        "excluded_train_sentence_hashes":
            len({
                r["normalized_sentence_sha256"]
                for r in excluded
            }),

        "exclusion_reasons":
            dict(
                Counter(
                    reason
                    for row in excluded
                    for reason
                    in row["firewall_reasons"]
                )
            ),

        "remaining_cross_split_normalized_overlap": {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        },
    }

    return (
        safe_train,
        list(dev_rows),
        list(test_rows),
        excluded,
        report,
    )
