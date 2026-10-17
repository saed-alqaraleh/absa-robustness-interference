
from absa_xfer.firewall import (
    apply_content_firewall,
)


def row(split, text, sid):
    return {
        "split": split,
        "sentence": text,
        "source_instance_id": sid,
        "cluster_id": sid,
        "label": "neutral",
    }


def test_train_test_content_duplicate_is_removed():

    train = [
        row(
            "train",
            "Same text",
            "train-a",
        ),
        row(
            "train",
            "Other text",
            "train-b",
        ),
    ]

    dev = [
        row(
            "dev",
            "Development sentence",
            "dev-a",
        )
    ]

    test = [
        row(
            "test",
            " same   TEXT ",
            "test-a",
        )
    ]

    (
        safe_train,
        _,
        _,
        excluded,
        report,
    ) = apply_content_firewall(
        train,
        dev,
        test,
    )

    assert [
        x["source_instance_id"]
        for x in safe_train
    ] == [
        "train-b"
    ]

    assert (
        excluded[0]
        ["firewall_reasons"]
        ==
        [
            "duplicate_of_protected_test"
        ]
    )

    assert (
        report[
            "excluded_train_rows"
        ]
        == 1
    )
