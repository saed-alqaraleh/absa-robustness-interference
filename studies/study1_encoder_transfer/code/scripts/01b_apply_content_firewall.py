
#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from absa_xfer.firewall import (
    apply_content_firewall,
)

from absa_xfer.integrity import (
    integrity_report,
)

from absa_xfer.io_utils import (
    read_jsonl,
    write_jsonl,
    atomic_write_json,
)

from absa_xfer.hashing import (
    hash_object,
    sha256_file,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--domain",
        required=True,
        choices=[
            "restaurant",
            "laptop",
        ],
    )

    ap.add_argument(
        "--data-dir",
        required=True,
    )

    ap.add_argument(
        "--manifest-copy",
        required=True,
    )

    ap.add_argument(
        "--audit-out",
        required=True,
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    backup_dir = (
        data_dir /
        "pre_firewall"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = [
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "split_manifest.json",
    ]

    # ------------------------------------------------------------
    # Immutable backup of the original prepared split.
    # ------------------------------------------------------------

    for name in required:

        source = data_dir / name
        backup = backup_dir / name

        if not source.exists():
            raise FileNotFoundError(
                source
            )

        if not backup.exists():
            shutil.copy2(
                source,
                backup,
            )

    # Always regenerate canonical data from
    # the immutable pre-firewall copy.
    train = read_jsonl(
        backup_dir /
        "train.jsonl"
    )

    dev = read_jsonl(
        backup_dir /
        "dev.jsonl"
    )

    test = read_jsonl(
        backup_dir /
        "test.jsonl"
    )

    (
        train_safe,
        dev_safe,
        test_safe,
        excluded,
        firewall_report,
    ) = apply_content_firewall(
        train,
        dev,
        test,
    )

    # ------------------------------------------------------------
    # Write canonical leakage-safe files.
    # ------------------------------------------------------------

    write_jsonl(
        data_dir /
        "train.jsonl",
        train_safe,
    )

    write_jsonl(
        data_dir /
        "dev.jsonl",
        dev_safe,
    )

    write_jsonl(
        data_dir /
        "test.jsonl",
        test_safe,
    )

    write_jsonl(
        data_dir /
        "firewall_exclusions.jsonl",
        excluded,
    )

    # ------------------------------------------------------------
    # Update split manifest while preserving ancestry.
    # ------------------------------------------------------------

    original_manifest = json.loads(
        (
            backup_dir /
            "split_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    parent_hash = (
        original_manifest.get(
            "manifest_hash"
        )
    )

    new_manifest = dict(
        original_manifest
    )

    new_manifest.pop(
        "manifest_hash",
        None,
    )

    new_manifest["counts"] = {
        "train":
            len(train_safe),

        "dev":
            len(dev_safe),

        "test":
            len(test_safe),
    }

    new_manifest[
        "parent_pre_firewall_manifest_hash"
    ] = parent_hash

    new_manifest[
        "content_firewall"
    ] = firewall_report

    new_manifest[
        "canonical_files"
    ] = {
        "train_sha256":
            sha256_file(
                data_dir /
                "train.jsonl"
            ),

        "dev_sha256":
            sha256_file(
                data_dir /
                "dev.jsonl"
            ),

        "test_sha256":
            sha256_file(
                data_dir /
                "test.jsonl"
            ),

        "exclusions_sha256":
            sha256_file(
                data_dir /
                "firewall_exclusions.jsonl"
            ),
    }

    new_manifest[
        "manifest_hash"
    ] = hash_object(
        new_manifest
    )

    atomic_write_json(
        data_dir /
        "split_manifest.json",
        new_manifest,
    )

    manifest_copy = Path(
        args.manifest_copy
    )

    manifest_copy.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        data_dir /
        "split_manifest.json",
        manifest_copy,
    )

    # ------------------------------------------------------------
    # Final integrity gate.
    # ------------------------------------------------------------

    report = integrity_report(
        train_safe,
        dev_safe,
        test_safe,
    )

    audit = {
        "domain":
            args.domain,

        "firewall":
            firewall_report,

        "excluded_records":
            excluded,

        "integrity":
            report,

        "canonical_split_manifest_hash":
            new_manifest[
                "manifest_hash"
            ],
    }

    atomic_write_json(
        args.audit_out,
        audit,
    )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    if report["blocking"]:
        raise RuntimeError(
            "Canonical split still "
            "fails integrity gate."
        )


if __name__ == "__main__":
    main()
