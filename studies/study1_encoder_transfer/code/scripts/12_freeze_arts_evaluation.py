#!/usr/bin/env python3
"""Freeze the official ARTS protected evaluation resources and re-preflight the minimal campaign.

This script is deliberately fail-closed. It never launches scientific training.
It expects the project state produced through Step 8 and the official ARTS repository
at the pinned commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

ROOT = Path("./data")
CODE = ROOT / "code"
SRC = CODE / "src"
ARTS_REPO = ROOT / "data_raw" / "external_repos" / "ARTS_TestSet"
DISCOVERY = ROOT / "audits" / "ARTS_DISCOVERY_REPORT.json"
PINNED_ARTS_COMMIT = "bb725433ec49a89cfed221e8b0b72a6a1053282a"

OFFICIAL_FILES = {
    "restaurant": ARTS_REPO / "data" / "arts_testset" / "rest_test_enriched.json",
    "laptop": ARTS_REPO / "data" / "arts_testset" / "laptop_test_enriched.json",
}

# Published ARTS family counts. The total enriched file also contains the clean source records.
EXPECTED = {
    "restaurant": {"Clean": 1120, "RevTgt": 846, "RevNon": 444, "AddDiff": 1120, "Enriched": 3530},
    "laptop": {"Clean": 638, "RevTgt": 466, "RevNon": 135, "AddDiff": 638, "Enriched": 1877},
}

SUFFIX_TO_FAMILY = {"_adv1": "RevTgt", "_adv2": "RevNon", "_adv3": "AddDiff"}
FAMILIES = ("RevTgt", "RevNon", "AddDiff")
VALID_POLARITIES = {"positive", "negative", "neutral"}
REV_POLARITY = {"positive": "negative", "negative": "positive", "neutral": "neutral"}

TEXT_FIELDS = ("sentence", "text", "context")
TERM_FIELDS = ("term", "aspect", "aspect_term", "target")
LABEL_FIELDS = ("polarity", "label", "sentiment")
START_FIELDS = ("from", "start", "aspect_start", "aspect_from")
END_FIELDS = ("to", "end", "aspect_end", "aspect_to")
ID_FIELDS = ("id", "example_id", "uid", "record_id")
SENTENCE_ID_FIELDS = ("sentence_id", "sentenceId", "sent_id", "sid")


def say(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    atomic_write_text(path, text)


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key detected: {key!r}")
        out[key] = value
    return out


def load_json_no_duplicates(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=_no_duplicate_pairs)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line, object_pairs_hook=_no_duplicate_pairs)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve git HEAD for {repo}: {exc}") from exc


def classify_record_key(key: str) -> tuple[str, str]:
    for suffix, family in SUFFIX_TO_FAMILY.items():
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if not base:
                raise ValueError(f"Malformed ARTS record key: {key!r}")
            return base, family
    if re.search(r"_adv\d+$", key):
        raise ValueError(f"Unknown ARTS adversarial suffix in record key: {key!r}")
    return key, "Clean"


def exact_term_spans(sentence: str, term: str) -> list[tuple[int, int]]:
    """Return every exact character span of ``term`` in ``sentence``.

    ARTS is an aspect-term classification benchmark.  Some released adversarial
    records contain stale/incorrect ``from``/``to`` offsets even though the
    sentence, target term, label, and source-id lineage are usable.  We never
    silently trust those derivative offsets; operational spans are resolved
    deterministically from the clean source target occurrence.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = sentence.find(term, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(term)))
        start = idx + 1
    return spans


def _span_is_exact(rec: dict[str, Any]) -> bool:
    start, end = rec["from"], rec["to"]
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(rec["sentence"])
        and rec["sentence"][start:end] == rec["term"]
    )


def validate_official_record(record_key: str, rec: dict[str, Any]) -> None:
    required = ("sentence", "term", "polarity", "id", "from", "to")
    missing = [k for k in required if k not in rec]
    if missing:
        raise ValueError(f"{record_key}: missing required keys {missing}")
    if not isinstance(rec["sentence"], str) or not rec["sentence"]:
        raise ValueError(f"{record_key}: invalid sentence")
    if not isinstance(rec["term"], str) or not rec["term"]:
        raise ValueError(f"{record_key}: invalid target term")
    if rec["polarity"] not in VALID_POLARITIES:
        raise ValueError(f"{record_key}: invalid polarity {rec['polarity']!r}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError(f"{record_key}: invalid source id")
    if not isinstance(rec["from"], int) or not isinstance(rec["to"], int):
        raise ValueError(f"{record_key}: from/to must be integers")

    _, family = classify_record_key(record_key)
    if family == "Clean":
        if not (0 <= rec["from"] < rec["to"] <= len(rec["sentence"])):
            raise ValueError(f"{record_key}: invalid clean target span [{rec['from']}, {rec['to']})")
        observed = rec["sentence"][rec["from"] : rec["to"]]
        if observed != rec["term"]:
            raise ValueError(
                f"{record_key}: clean target-span mismatch: "
                f"sentence[from:to]={observed!r}, term={rec['term']!r}"
            )


def classify_term_relation(source_term: str, variant_term: str) -> str:
    """Describe the literal relation between released source/variant target strings.

    ARTS source linkage is defined by the record key / ``id`` field.  The released
    adversarial target string is not guaranteed to be character-identical to the
    clean source target string (for example, ``warm pitas`` -> ``pitas``).  We
    therefore preserve the released variant term and audit, rather than normalize,
    term changes.
    """
    if variant_term == source_term:
        return "same"
    if variant_term in source_term:
        return "variant_substring_of_source"
    if source_term in variant_term:
        return "source_substring_of_variant"
    return "different"


def _source_target_ordinal(source: dict[str, Any], record_key: str) -> tuple[int, list[tuple[int, int]]]:
    source_spans = exact_term_spans(source["sentence"], source["term"])
    source_target = (source["from"], source["to"])
    ordinals = [i for i, span in enumerate(source_spans) if span == source_target]
    if len(ordinals) != 1:
        raise ValueError(
            f"{record_key}: clean source target occurrence is not uniquely identifiable: "
            f"source_span={source_target}, occurrences={source_spans}"
        )
    return ordinals[0], source_spans


def audit_variant_target_metadata(
    record_key: str,
    source: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    """Audit derivative target metadata without modifying the official ARTS record.

    Scientific policy (V4): official protected evaluation content is immutable.
    The classifier used by this project consumes ``sentence`` + ``aspect_term``
    and does not consume character offsets. Therefore derivative ``sentence``,
    ``term``, ``polarity``, ``from`` and ``to`` are preserved exactly as released.

    ARTS contains a small number of derivative metadata inconsistencies (for
    example stale offsets, source/variant term changes, or spelling correction
    in the sentence while the released term remains unchanged). These are
    provenance/audit findings, not grounds for silently rewriting a protected
    benchmark. Source linkage and family-label semantics remain strict gates.
    """
    source_ordinal, source_spans = _source_target_ordinal(source, record_key)
    source_term = source["term"]
    variant_term = variant["term"]
    relation = classify_term_relation(source_term, variant_term)
    variant_spans = exact_term_spans(variant["sentence"], variant_term)

    start, end = variant["from"], variant["to"]
    span_in_bounds = 0 <= start < end <= len(variant["sentence"])
    span_surface = variant["sentence"][start:end] if span_in_bounds else None
    official_exact = span_in_bounds and span_surface == variant_term

    if official_exact:
        status = "released_exact"
    elif span_in_bounds and variant_spans:
        status = "released_span_mismatch_term_present"
    elif span_in_bounds and not variant_spans:
        status = "released_span_surface_differs_term_absent"
    elif (not span_in_bounds) and variant_spans:
        status = "released_span_out_of_bounds_term_present"
    else:
        status = "released_span_out_of_bounds_term_absent"

    # Diagnostic-only recoverable span. It is NEVER substituted into the
    # protected evaluation record used by the model.
    diagnostic_span = None
    diagnostic_rule = None
    if official_exact:
        diagnostic_span = (start, end)
        diagnostic_rule = "released_exact"
    elif len(variant_spans) == 1:
        diagnostic_span = variant_spans[0]
        diagnostic_rule = "unique_released_term_occurrence"
    elif variant_term == source_term and source_ordinal < len(variant_spans):
        diagnostic_span = variant_spans[source_ordinal]
        diagnostic_rule = "source_occurrence_ordinal"
    elif span_in_bounds:
        diagnostic_span = (start, end)
        diagnostic_rule = "released_span_surface_only"

    return {
        # Raw official derivative metadata, repeated explicitly for provenance.
        "arts_official_from": start,
        "arts_official_to": end,
        "arts_official_span_in_bounds": span_in_bounds,
        "arts_official_span_exact": official_exact,
        "arts_released_span_surface": span_surface,
        "arts_source_target_ordinal": source_ordinal,
        "arts_source_term": source_term,
        "arts_variant_term": variant_term,
        "arts_target_term_changed": variant_term != source_term,
        "arts_term_relation": relation,
        "arts_source_term_occurrences": len(source_spans),
        "arts_variant_term_occurrences": len(variant_spans),
        "arts_variant_term_present_in_sentence": bool(variant_spans),
        "arts_span_status": status,
        "arts_diagnostic_resolved_from": None if diagnostic_span is None else diagnostic_span[0],
        "arts_diagnostic_resolved_to": None if diagnostic_span is None else diagnostic_span[1],
        "arts_diagnostic_resolution_rule": diagnostic_rule,
        "arts_processed_metadata_policy": "preserve_official_derivative_verbatim",
    }

def split_official_records(data: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = {"Clean": [], **{f: [] for f in FAMILIES}}
    clean_by_id: dict[str, dict[str, Any]] = {}

    for record_key, rec in data.items():
        if not isinstance(rec, dict):
            raise ValueError(f"{record_key}: expected object, got {type(rec).__name__}")
        validate_official_record(record_key, rec)
        base, family = classify_record_key(record_key)
        if family == "Clean":
            if rec["id"] != record_key:
                raise ValueError(f"{record_key}: clean record id must equal outer record key")
            if base in clean_by_id:
                raise ValueError(f"Duplicate clean source id: {base}")
            clean_by_id[base] = rec

    seen_family_source: set[tuple[str, str]] = set()
    for record_key, rec in data.items():
        base, family = classify_record_key(record_key)
        source = clean_by_id.get(base)
        if source is None:
            raise ValueError(f"{record_key}: source clean record {base!r} is missing")
        if rec["id"] != base:
            raise ValueError(f"{record_key}: record id {rec['id']!r} does not point to source {base!r}")
        if family != "Clean":
            key = (family, base)
            if key in seen_family_source:
                raise ValueError(f"More than one {family} derivative for source {base}")
            seen_family_source.add(key)
            if rec["sentence"] == source["sentence"]:
                raise ValueError(f"{record_key}: derivative sentence is identical to source sentence")
            if family == "RevTgt" and rec["polarity"] != REV_POLARITY[source["polarity"]]:
                raise ValueError(
                    f"{record_key}: RevTgt label rule failed: {source['polarity']} -> {rec['polarity']}"
                )
            if family in {"RevNon", "AddDiff"} and rec["polarity"] != source["polarity"]:
                raise ValueError(
                    f"{record_key}: {family} must preserve target polarity: "
                    f"{source['polarity']} != {rec['polarity']}"
                )

        enriched = dict(rec)
        enriched["arts_record_id"] = record_key
        enriched["arts_source_id"] = base
        enriched["arts_family"] = family
        if family == "Clean":
            enriched.update({
                "arts_official_from": rec["from"],
                "arts_official_to": rec["to"],
                "arts_official_span_in_bounds": True,
                "arts_official_span_exact": True,
                "arts_released_span_surface": rec["term"],
                "arts_source_target_ordinal": exact_term_spans(rec["sentence"], rec["term"]).index(
                    (rec["from"], rec["to"])
                ),
                "arts_source_term": rec["term"],
                "arts_variant_term": rec["term"],
                "arts_target_term_changed": False,
                "arts_term_relation": "same",
                "arts_source_term_occurrences": len(exact_term_spans(rec["sentence"], rec["term"])),
                "arts_variant_term_occurrences": len(exact_term_spans(rec["sentence"], rec["term"])),
                "arts_variant_term_present_in_sentence": True,
                "arts_span_status": "released_clean_exact",
                "arts_diagnostic_resolved_from": rec["from"],
                "arts_diagnostic_resolved_to": rec["to"],
                "arts_diagnostic_resolution_rule": "released_clean_exact",
                "arts_processed_metadata_policy": "preserve_official_clean_verbatim",
            })
        else:
            enriched.update(audit_variant_target_metadata(record_key, source, rec))

        families[family].append(enriched)

    return families


def summarize_span_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    inexact: list[str] = []
    out_of_bounds: list[str] = []
    term_changed: list[str] = []
    term_absent: list[str] = []
    term_relations: dict[str, int] = {}
    for rec in rows:
        status = rec["arts_span_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if not rec["arts_official_span_exact"]:
            inexact.append(rec["arts_record_id"])
        if not rec["arts_official_span_in_bounds"]:
            out_of_bounds.append(rec["arts_record_id"])
        if rec.get("arts_target_term_changed", False):
            term_changed.append(rec["arts_record_id"])
        if not rec.get("arts_variant_term_present_in_sentence", False):
            term_absent.append(rec["arts_record_id"])
        rel = rec.get("arts_term_relation", "unknown")
        term_relations[rel] = term_relations.get(rel, 0) + 1
    return {
        "n_records": len(rows),
        "status_counts": status_counts,
        "n_official_span_inexact": len(inexact),
        "n_official_span_out_of_bounds": len(out_of_bounds),
        "n_target_term_changed": len(term_changed),
        "n_variant_term_absent_from_sentence": len(term_absent),
        "term_relation_counts": term_relations,
        "official_inexact_record_ids": inexact,
        "official_out_of_bounds_record_ids": out_of_bounds,
        "target_term_changed_record_ids": term_changed,
        "variant_term_absent_record_ids": term_absent,
    }


@dataclass(frozen=True)
class LocalSchema:
    text: str
    term: str
    label: str
    start: str | None
    end: str | None
    record_id: str | None
    sentence_id: str | None = None


def _first_present(row: dict[str, Any], candidates: tuple[str, ...], required: bool = True) -> str | None:
    found = [x for x in candidates if x in row]
    if len(found) > 1:
        # Multiple aliases are allowed only if they carry identical information across the project;
        # choosing one silently would make lineage ambiguous, so fail closed.
        raise ValueError(f"Ambiguous schema: multiple candidate fields present: {found}")
    if not found:
        if required:
            raise ValueError(f"Missing required field; expected one of {candidates}")
        return None
    return found[0]


def detect_local_schema(rows: list[dict[str, Any]]) -> LocalSchema:
    if not rows:
        raise ValueError("Local clean test JSONL is empty")
    first = rows[0]
    text = _first_present(first, TEXT_FIELDS)
    term = _first_present(first, TERM_FIELDS)
    label = _first_present(first, LABEL_FIELDS)
    start = _first_present(first, START_FIELDS, required=False)
    end = _first_present(first, END_FIELDS, required=False)
    record_id = _first_present(first, ID_FIELDS, required=False)
    sentence_id = _first_present(first, SENTENCE_ID_FIELDS, required=False)
    if (start is None) != (end is None):
        raise ValueError("Local schema has only one of start/end span fields")
    schema = LocalSchema(text, term, label, start, end, record_id, sentence_id)
    required_fields = [text, term, label] + ([start, end] if start and end else []) + ([sentence_id] if sentence_id else [])
    for i, row in enumerate(rows):
        missing = [x for x in required_fields if x not in row]
        if missing:
            raise ValueError(f"Local row {i} is missing schema fields {missing}")
    return schema


def _local_match_key(row: dict[str, Any], schema: LocalSchema) -> tuple[Any, ...]:
    if schema.start and schema.end:
        return (row[schema.text], row[schema.term], int(row[schema.start]), int(row[schema.end]))
    return (row[schema.text], row[schema.term])


def _official_match_key(rec: dict[str, Any], schema: LocalSchema) -> tuple[Any, ...]:
    if schema.start and schema.end:
        return (rec["sentence"], rec["term"], int(rec["from"]), int(rec["to"]))
    return (rec["sentence"], rec["term"])


def arts_source_sentence_id(arts_source_id: str) -> str | None:
    """Recover the original SemEval sentence id encoded in an ARTS clean/source id.

    ARTS source ids append an underscore plus an aspect-instance counter to the
    SemEval sentence id, e.g. ``456:1_0`` -> ``456:1`` and
    ``11302355#533813#0_1`` -> ``11302355#533813#0``.
    """
    m = re.match(r"^(.*)_([0-9]+)$", str(arts_source_id))
    return m.group(1) if m else None


def _literal_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    if not term:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(term), text)]


def _target_occurrence_ordinal(sentence: str, term: str, start: int, end: int) -> int | None:
    occurrences = _literal_occurrences(sentence, term)
    for idx, span in enumerate(occurrences):
        if span == (int(start), int(end)):
            return idx
    # If there is only one literal target occurrence, ARTS text cleanup may have
    # shifted the released clean offsets while target identity remains unambiguous.
    if len(occurrences) == 1:
        return 0
    return None


def _candidate_summary(row: dict[str, Any], schema: LocalSchema) -> dict[str, Any]:
    out = {
        "sentence_id": row.get(schema.sentence_id) if schema.sentence_id else None,
        "term": row.get(schema.term),
        "label": row.get(schema.label),
        "source_instance_id": row.get("source_instance_id"),
    }
    if schema.start and schema.end:
        out["span"] = [int(row[schema.start]), int(row[schema.end])]
    return out


def align_official_clean_to_local(
    official_clean: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
    schema: LocalSchema,
    failure_report_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Align ARTS clean/source records to canonical local SemEval test rows.

    Preferred lineage uses the SemEval sentence id encoded in each ARTS source id.
    Exact source text is *audited*, not required, because the released ARTS clean
    resources contain documented text cleanups (e.g. inserted whitespace) that can
    shift character offsets without changing source identity.

    Within a sentence id we require the same target term and polarity. Ambiguity is
    resolved first by exact span and then by target-occurrence ordinal. A unique
    term+polarity candidate is also safe because sentence id is exact. Anything
    unresolved is collected and reported in one pass before failing closed.
    """
    legacy_index: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in local_rows:
        legacy_index.setdefault(_local_match_key(row, schema), []).append(row)

    by_sentence_id: dict[str, list[dict[str, Any]]] = {}
    if schema.sentence_id:
        for row in local_rows:
            by_sentence_id.setdefault(str(row[schema.sentence_id]), []).append(row)

    mapping: dict[str, dict[str, Any]] = {}
    label_sets: dict[str, set[Any]] = {p: set() for p in VALID_POLARITIES}
    used_local_ids: set[int] = set()
    failures: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    text_diff_ids: list[str] = []
    offset_diff_ids: list[str] = []

    for rec in official_clean:
        selected: dict[str, Any] | None = None
        mode: str | None = None
        arts_sid = arts_source_sentence_id(rec["arts_source_id"])

        if schema.sentence_id and arts_sid is not None:
            sentence_candidates = by_sentence_id.get(arts_sid, [])
            candidates = [
                row for row in sentence_candidates
                if row.get(schema.term) == rec["term"]
            ]
            # The local label can have a non-string encoding, so once a label map
            # is known we cannot use it yet. Under this project the canonical rows
            # use textual labels; when they do, use polarity to strengthen lineage.
            same_label = [row for row in candidates if str(row.get(schema.label)).casefold() == rec["polarity"]]
            if same_label:
                candidates = same_label

            if len(candidates) > 1 and schema.start and schema.end:
                exact_span = [
                    row for row in candidates
                    if int(row[schema.start]) == int(rec["from"])
                    and int(row[schema.end]) == int(rec["to"])
                ]
                if len(exact_span) == 1:
                    selected = exact_span[0]
                    mode = "sentence_id_exact_span"

            if selected is None and len(candidates) > 1 and schema.start and schema.end:
                official_ord = _target_occurrence_ordinal(
                    rec["sentence"], rec["term"], int(rec["from"]), int(rec["to"])
                )
                if official_ord is not None:
                    ordinal_matches = []
                    for row in candidates:
                        local_ord = _target_occurrence_ordinal(
                            str(row[schema.text]), str(row[schema.term]),
                            int(row[schema.start]), int(row[schema.end]),
                        )
                        if local_ord == official_ord:
                            ordinal_matches.append(row)
                    if len(ordinal_matches) == 1:
                        selected = ordinal_matches[0]
                        mode = "sentence_id_occurrence_ordinal"

            if selected is None and len(candidates) == 1:
                selected = candidates[0]
                mode = "sentence_id_unique_term_label"

            if selected is None:
                failures.append({
                    "arts_source_id": rec["arts_source_id"],
                    "arts_sentence_id": arts_sid,
                    "term": rec["term"],
                    "polarity": rec["polarity"],
                    "official_span": [int(rec["from"]), int(rec["to"])],
                    "official_sentence": rec["sentence"],
                    "same_sentence_id_local_rows": len(sentence_candidates),
                    "term_label_candidates": [_candidate_summary(x, schema) for x in candidates[:20]],
                    "reason": "sentence-id lineage did not resolve to exactly one target row",
                })
                continue
        else:
            key = _official_match_key(rec, schema)
            candidates = legacy_index.get(key, [])
            if len(candidates) == 1:
                selected = candidates[0]
                mode = "legacy_exact_content"
            else:
                failures.append({
                    "arts_source_id": rec["arts_source_id"],
                    "arts_sentence_id": arts_sid,
                    "term": rec["term"],
                    "polarity": rec["polarity"],
                    "official_span": [int(rec["from"]), int(rec["to"])],
                    "official_sentence": rec["sentence"],
                    "legacy_candidate_count": len(candidates),
                    "reason": "no sentence_id field available and exact-content lineage was not one-to-one",
                })
                continue

        assert selected is not None and mode is not None
        oid = id(selected)
        if oid in used_local_ids:
            failures.append({
                "arts_source_id": rec["arts_source_id"],
                "arts_sentence_id": arts_sid,
                "term": rec["term"],
                "polarity": rec["polarity"],
                "reason": "local row would be reused by multiple ARTS clean sources",
                "selected_local": _candidate_summary(selected, schema),
            })
            continue

        used_local_ids.add(oid)
        mapping[rec["arts_source_id"]] = selected
        label_sets[rec["polarity"]].add(selected[schema.label])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if str(selected[schema.text]) != rec["sentence"]:
            text_diff_ids.append(rec["arts_source_id"])
        if schema.start and schema.end and (
            int(selected[schema.start]) != int(rec["from"])
            or int(selected[schema.end]) != int(rec["to"])
        ):
            offset_diff_ids.append(rec["arts_source_id"])

    unaligned_local = [
        _candidate_summary(row, schema) for row in local_rows if id(row) not in used_local_ids
    ]
    if failures or len(mapping) != len(local_rows) or unaligned_local:
        diagnostic = {
            "status": "FAIL_CLOSED",
            "official_clean_count": len(official_clean),
            "local_clean_count": len(local_rows),
            "aligned_count": len(mapping),
            "failure_count": len(failures),
            "sentence_id_field": schema.sentence_id,
            "failures": failures,
            "unaligned_local_count": len(unaligned_local),
            "unaligned_local_rows": unaligned_local[:100],
        }
        if failure_report_path is not None:
            atomic_write_json(failure_report_path, diagnostic)
            where = f" See {failure_report_path}"
        else:
            where = ""
        raise ValueError(
            f"Clean source-lineage audit failed collectively: aligned={len(mapping)}/"
            f"{len(official_clean)}, failures={len(failures)}, "
            f"unaligned_local={len(unaligned_local)}.{where}"
        )

    label_map: dict[str, Any] = {}
    for polarity in sorted(VALID_POLARITIES):
        values = label_sets[polarity]
        if len(values) != 1:
            raise ValueError(f"Cannot infer unique local encoding for {polarity}: {values!r}")
        label_map[polarity] = next(iter(values))

    audit = {
        "alignment_policy": "SemEval sentence-id lineage + target identity; released ARTS clean text differences audited, not normalized",
        "sentence_id_field": schema.sentence_id,
        "aligned_count": len(mapping),
        "mode_counts": mode_counts,
        "n_clean_text_differences": len(text_diff_ids),
        "clean_text_difference_source_ids": text_diff_ids,
        "n_clean_offset_differences": len(offset_diff_ids),
        "clean_offset_difference_source_ids": offset_diff_ids,
    }
    return mapping, label_map, audit


def adapt_variant_to_local(
    source_local: dict[str, Any],
    variant: dict[str, Any],
    schema: LocalSchema,
    label_map: dict[str, Any],
) -> dict[str, Any]:
    out = dict(source_local)
    out[schema.text] = variant["sentence"]
    out[schema.term] = variant["term"]
    out[schema.label] = label_map[variant["polarity"]]
    if schema.start and schema.end:
        out[schema.start] = variant["from"]
        out[schema.end] = variant["to"]
    if schema.record_id:
        out[schema.record_id] = variant["arts_record_id"]
    out["arts_record_id"] = variant["arts_record_id"]
    out["arts_source_id"] = variant["arts_source_id"]
    out["arts_family"] = variant["arts_family"]
    out["arts_repo_commit"] = PINNED_ARTS_COMMIT
    for key in (
        "arts_official_from", "arts_official_to", "arts_official_span_in_bounds",
        "arts_official_span_exact", "arts_released_span_surface",
        "arts_source_target_ordinal", "arts_source_term", "arts_variant_term",
        "arts_target_term_changed", "arts_term_relation",
        "arts_source_term_occurrences", "arts_variant_term_occurrences",
        "arts_variant_term_present_in_sentence", "arts_span_status",
        "arts_diagnostic_resolved_from", "arts_diagnostic_resolved_to",
        "arts_diagnostic_resolution_rule", "arts_processed_metadata_policy",
    ):
        if key in variant:
            out[key] = variant[key]
    return out


def normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def verify_discovery_hash(official_path: Path, discovery: dict[str, Any]) -> dict[str, Any]:
    rel = official_path.relative_to(ARTS_REPO).as_posix()
    matches = [x for x in discovery.get("files", []) if str(x.get("relative_path", "")).replace("\\", "/") == rel]
    if len(matches) != 1:
        raise ValueError(f"Discovery report must contain exactly one entry for {rel}; found {len(matches)}")
    current = sha256_file(official_path)
    expected = matches[0].get("sha256")
    if expected and current != expected:
        raise ValueError(f"Official ARTS file hash changed since discovery: {rel}")
    return {"relative_path": rel, "sha256": current, "size_bytes": official_path.stat().st_size}


def find_text_field(row: dict[str, Any]) -> str:
    fields = [x for x in TEXT_FIELDS if x in row]
    if len(fields) != 1:
        raise ValueError(f"Cannot identify exactly one text field in training row; found {fields}")
    return fields[0]


def verify_model_input_contract() -> dict[str, Any]:
    """Fail if the experiment runner starts consuming derivative character offsets.

    The benchmark-faithful V4 policy is valid because PairDataset tokenizes only
    sentence + aspect_term and uses label as supervision. If that implementation
    changes, derivative offset anomalies must be reconsidered before any run.
    """
    import ast

    path = SRC / "absa_xfer" / "training.py"
    if not path.exists():
        raise RuntimeError(f"Training module missing: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PairDataset":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__getitem__":
                    target = item
                    break
    if target is None:
        raise RuntimeError("Cannot locate PairDataset.__getitem__ in training.py")

    row_keys: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "r":
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                row_keys.add(sl.value)
    expected = {"sentence", "aspect_term", "label"}
    if row_keys != expected:
        raise RuntimeError(
            "PairDataset model-input contract changed; expected row keys "
            f"{sorted(expected)}, observed {sorted(row_keys)}. Reassess ARTS metadata policy."
        )
    return {
        "training_module": str(path),
        "training_module_sha256": sha256_file(path),
        "pair_dataset_row_keys": sorted(row_keys),
        "character_offsets_consumed": False,
        "policy": "official derivative from/to preserved verbatim as non-model metadata",
    }

def protected_text_firewall(protected_texts: set[str]) -> dict[str, Any]:
    materialized = ROOT / "transformations" / "materialized"
    train_files = sorted(materialized.glob("*/*/train.jsonl"))
    if len(train_files) < 12:
        raise ValueError(f"Expected at least 12 materialized train files, found {len(train_files)}")
    overlaps: list[dict[str, Any]] = []
    total_rows = 0
    for path in train_files:
        rows = load_jsonl(path)
        total_rows += len(rows)
        if not rows:
            continue
        field = find_text_field(rows[0])
        for i, row in enumerate(rows, 1):
            if field not in row or not isinstance(row[field], str):
                raise ValueError(f"{path}:{i}: invalid text field")
            n = normalized_text(row[field])
            if n in protected_texts:
                overlaps.append({"file": str(path), "line": i, "text_sha256": hashlib.sha256(n.encode()).hexdigest()})
                if len(overlaps) >= 20:
                    break
        if len(overlaps) >= 20:
            break
    if overlaps:
        raise ValueError(
            "Protected ARTS sentence overlap detected in materialized training data. "
            f"First overlaps: {overlaps[:5]}"
        )
    return {"train_files_checked": len(train_files), "train_rows_checked": total_rows, "overlap_count": 0}


def campaign_preflight() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    out = ROOT / "campaigns" / "minimal_decisive"
    cmd = [
        "/usr/bin/python3",
        "scripts/11_build_campaign.py",
        "--config", "configs/base.yaml",
        "--stage", "minimal",
        "--out-dir", str(out),
        "--model-pins", str(ROOT / "manifests" / "model_pins.json"),
    ]
    proc = subprocess.run(cmd, cwd=CODE, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Campaign generation failed:\n{proc.stdout}\n{proc.stderr}")

    plan = out / "minimal_plan.csv"
    commands_path = out / "minimal_commands.sh"
    with plan.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 36:
        raise ValueError(f"Expected 36 campaign rows, found {len(rows)}")
    run_ids = [r.get("run_id", "") for r in rows]
    if any(not x for x in run_ids) or len(set(run_ids)) != 36:
        raise ValueError("Campaign must contain 36 nonempty unique run IDs")

    commands = [
        line.strip() for line in commands_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("set ")
    ]
    if len(commands) != 36:
        raise ValueError(f"Expected 36 executable commands, found {len(commands)}")

    command_signatures = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in commands]
    if len(set(command_signatures)) != 36:
        raise ValueError("Campaign contains duplicate command signatures")

    required_global = [
        "/usr/bin/python3", "/transformations/materialized/",
        "FINAL_GENERATION_VALID_SOURCE_SELECTION.json", "--eval Clean=",
        "--eval RevTgt=", "--eval RevNon=", "--eval AddDiff=",
        "--pair-order sentence_first", "--repo-root", "--model-revision", "--tokenizer-revision",
    ]
    problems: list[dict[str, Any]] = []
    for i, line in enumerate(commands, 1):
        missing = [frag for frag in required_global if frag not in line]
        for domain in ("restaurant", "laptop"):
            if f"--dataset {domain}" in line:
                expected_eval = {
                    "Clean": ROOT / "data_processed" / domain / "test.jsonl",
                    "RevTgt": ROOT / "data_processed" / domain / "evaluation" / "RevTgt.jsonl",
                    "RevNon": ROOT / "data_processed" / domain / "evaluation" / "RevNon.jsonl",
                    "AddDiff": ROOT / "data_processed" / domain / "evaluation" / "AddDiff.jsonl",
                }
                for family, path in expected_eval.items():
                    exact = f"--eval {family}={path}"
                    if exact not in line:
                        missing.append(exact)
                    if not path.is_file():
                        missing.append(f"MISSING_FILE:{path}")

                tokens = shlex.split(line)
                for flag in ("--train", "--dev", "--split-manifest", "--transformation-manifest", "--source-selection", "--repo-root"):
                    try:
                        value = tokens[tokens.index(flag) + 1]
                    except (ValueError, IndexError):
                        missing.append(f"MISSING_FLAG:{flag}")
                        continue
                    path = Path(value)
                    exists = path.is_dir() if flag == "--repo-root" else path.is_file()
                    if not exists:
                        missing.append(f"MISSING_FILE:{flag}={path}")
        if missing:
            problems.append({"index": i, "missing": missing, "command": line})
    if problems:
        raise ValueError(f"{len(problems)} campaign commands failed final ARTS-path preflight: {problems[:2]}")

    return {
        "plan": str(plan),
        "commands": str(commands_path),
        "n_runs": 36,
        "unique_run_ids": 36,
        "unique_command_signatures": 36,
        "command_signatures_sha256": hashlib.sha256("\n".join(command_signatures).encode()).hexdigest(),
        "status": "READY_NOT_LAUNCHED",
    }


def main() -> None:
    say("STEP 9 — FREEZE OFFICIAL ARTS EVALUATION + FINAL MINIMAL-CAMPAIGN PREFLIGHT")

    for path in (ROOT, CODE, ARTS_REPO, DISCOVERY):
        if not path.exists():
            raise RuntimeError(f"Required project artifact missing: {path}")

    head = git_head(ARTS_REPO)
    if head != PINNED_ARTS_COMMIT:
        raise RuntimeError(f"ARTS commit mismatch: {head} != {PINNED_ARTS_COMMIT}")
    print("ARTS commit: PASS", head)

    discovery = load_json_no_duplicates(DISCOVERY)
    mapping: dict[str, Any] = {
        "status": "FROZEN_VALIDATED",
        "arts_repo": str(ARTS_REPO),
        "arts_commit": head,
        "record_key_mapping": {"no_suffix": "Clean", "_adv1": "RevTgt", "_adv2": "RevNon", "_adv3": "AddDiff"},
        "domains": {},
    }
    manifest: dict[str, Any] = {
        "status": "FROZEN_VALIDATED",
        "arts_commit": head,
        "discovery_report": str(DISCOVERY),
        "discovery_report_sha256": sha256_file(DISCOVERY),
        "domains": {},
    }
    report_lines = [
        "# ARTS Evaluation Integrity Report",
        "",
        f"- ARTS commit: `{head}`",
        "- Family mapping: unsuffixed=Clean, `_adv1`=RevTgt, `_adv2`=RevNon, `_adv3`=AddDiff.",
        "- Status: **PASS** only if this script reaches the final gate.",
        "",
    ]

    say("MODEL INPUT CONTRACT")
    model_input_contract = verify_model_input_contract()
    manifest["model_input_contract"] = model_input_contract
    mapping["model_input_contract"] = model_input_contract
    print("PASS", model_input_contract)
    report_lines.extend([
        "## Model-input contract",
        "",
        "- `PairDataset` consumes only `sentence`, `aspect_term`, and `label`.",
        "- Character offsets are not consumed by the scientific classifier.",
        "- Therefore official ARTS derivative sentence/term/label/from/to are preserved verbatim; derivative metadata inconsistencies are audited rather than rewritten.",
        "",
    ])

    protected_texts: set[str] = set()

    for domain in ("restaurant", "laptop"):
        say(f"{domain.upper()} — OFFICIAL INGESTION")
        official_path = OFFICIAL_FILES[domain]
        if not official_path.exists():
            raise RuntimeError(f"Official ARTS file missing: {official_path}")
        raw_info = verify_discovery_hash(official_path, discovery)
        raw = load_json_no_duplicates(official_path)
        if not isinstance(raw, dict):
            raise ValueError(f"Official ARTS root must be an object: {official_path}")
        if len(raw) != EXPECTED[domain]["Enriched"]:
            raise ValueError(f"{domain}: enriched count {len(raw)} != {EXPECTED[domain]['Enriched']}")

        families = split_official_records(raw)
        counts = {family: len(rows) for family, rows in families.items()}
        for family in ("Clean", *FAMILIES):
            if counts[family] != EXPECTED[domain][family]:
                raise ValueError(f"{domain}/{family}: {counts[family]} != expected {EXPECTED[domain][family]}")
        if {x["arts_source_id"] for x in families["AddDiff"]} != {x["arts_source_id"] for x in families["Clean"]}:
            raise ValueError(f"{domain}: AddDiff is not one-to-one with the complete clean test cohort")

        for rows in families.values():
            for rec in rows:
                protected_texts.add(normalized_text(rec["sentence"]))

        local_test = ROOT / "data_processed" / domain / "test.jsonl"
        if not local_test.exists():
            raise RuntimeError(f"Local clean test file missing: {local_test}")
        local_rows = load_jsonl(local_test)
        if len(local_rows) != EXPECTED[domain]["Clean"]:
            raise ValueError(f"{domain}: local clean test count {len(local_rows)} != {EXPECTED[domain]['Clean']}")
        schema = detect_local_schema(local_rows)
        alignment_failure_report = ROOT / "audits" / "arts_mapping" / f"{domain}_CLEAN_ALIGNMENT_FAILURES.json"
        aligned, label_map, clean_alignment_audit = align_official_clean_to_local(
            families["Clean"], local_rows, schema, alignment_failure_report
        )

        out_dir = ROOT / "data_processed" / domain / "evaluation"
        output_info: dict[str, Any] = {}
        for family in FAMILIES:
            rows_out = [adapt_variant_to_local(aligned[v["arts_source_id"]], v, schema, label_map) for v in families[family]]
            out_path = out_dir / f"{family}.jsonl"
            atomic_write_jsonl(out_path, rows_out)
            reread = load_jsonl(out_path)
            if len(reread) != EXPECTED[domain][family]:
                raise ValueError(f"Post-write row-count mismatch for {out_path}")
            output_info[family] = {
                "path": str(out_path),
                "count": len(rows_out),
                "sha256": sha256_file(out_path),
                "unique_record_ids": len({r["arts_record_id"] for r in rows_out}),
                "unique_source_ids": len({r["arts_source_id"] for r in rows_out}),
            }
            if output_info[family]["unique_record_ids"] != len(rows_out):
                raise ValueError(f"{domain}/{family}: duplicate processed record IDs")
            if output_info[family]["unique_source_ids"] != len(rows_out):
                raise ValueError(f"{domain}/{family}: duplicate processed source IDs")

        span_audit = {family: summarize_span_audit(families[family]) for family in FAMILIES}
        domain_map = {
            "official_file": raw_info,
            "clean_local_path": str(local_test),
            "clean_local_sha256": sha256_file(local_test),
            "local_schema": schema.__dict__,
            "label_encoding": label_map,
            "clean_alignment_audit": clean_alignment_audit,
            "counts": counts,
            "span_audit": span_audit,
            "evaluation": output_info,
        }
        mapping["domains"][domain] = domain_map
        manifest["domains"][domain] = domain_map
        print("Counts:", counts)
        print("Local schema:", schema)
        print("Label map:", label_map)
        print("Clean alignment:", clean_alignment_audit)
        for family in FAMILIES:
            print(
                f"  {family:7s}: {output_info[family]['count']:4d}  "
                f"{output_info[family]['sha256'][:16]}...  "
                f"span_inexact={span_audit[family]['n_official_span_inexact']}  "
                f"term_changed={span_audit[family]['n_target_term_changed']}  "
                f"term_absent={span_audit[family]['n_variant_term_absent_from_sentence']}"
            )

        report_lines.extend([
            f"## {domain.capitalize()}",
            "",
            f"- Official enriched file: `{raw_info['relative_path']}`",
            f"- Official SHA-256: `{raw_info['sha256']}`",
            f"- Counts: Clean={counts['Clean']}, RevTgt={counts['RevTgt']}, RevNon={counts['RevNon']}, AddDiff={counts['AddDiff']}.",
            f"- Local clean cohort aligned one-to-one: {len(aligned)}/{len(local_rows)}.",
            f"- Clean lineage policy: `{clean_alignment_audit['alignment_policy']}`.",
            f"- Clean ARTS-vs-local text differences audited: {clean_alignment_audit['n_clean_text_differences']}.",
            f"- Clean ARTS-vs-local offset differences audited: {clean_alignment_audit['n_clean_offset_differences']}.",
            f"- Clean alignment modes: `{clean_alignment_audit['mode_counts']}`.",
            f"- Local label encoding inferred from aligned sources: `{label_map}`.",
            "- Official derivative `sentence`, `term`, `polarity`, `from`, and `to` are preserved verbatim in processed protected evaluation data.",
            "- Derivative span inconsistencies are audit metadata only because the frozen classifier consumes sentence + aspect term, not character offsets.",
            f"- Released derivative span mismatches: RevTgt={span_audit['RevTgt']['n_official_span_inexact']}, RevNon={span_audit['RevNon']['n_official_span_inexact']}, AddDiff={span_audit['AddDiff']['n_official_span_inexact']}.",
            f"- Released derivative term absent from its sentence: RevTgt={span_audit['RevTgt']['n_variant_term_absent_from_sentence']}, RevNon={span_audit['RevNon']['n_variant_term_absent_from_sentence']}, AddDiff={span_audit['AddDiff']['n_variant_term_absent_from_sentence']}.",
            f"- Released derivative target-term changes vs. clean source: RevTgt={span_audit['RevTgt']['n_target_term_changed']}, RevNon={span_audit['RevNon']['n_target_term_changed']}, AddDiff={span_audit['AddDiff']['n_target_term_changed']}. These released terms are preserved, not normalized.",
            "- Every derivative points to an existing released source ID; source lineage is established by the official record key/`id` link. Literal source/variant term equality is audited rather than assumed.",
            "- RevTgt target-label rule and RevNon/AddDiff target-label preservation checks passed.",
            "",
        ])

    say("PROTECTED-TEST FIREWALL")
    firewall = protected_text_firewall(protected_texts)
    manifest["protected_test_firewall"] = firewall
    mapping["protected_test_firewall"] = firewall
    report_lines.extend([
        "## Protected-test firewall",
        "",
        f"- Materialized train files checked: {firewall['train_files_checked']}.",
        f"- Training rows checked: {firewall['train_rows_checked']}.",
        "- Exact normalized sentence overlap with official ARTS Clean/RevTgt/RevNon/AddDiff: **0**.",
        "",
    ])
    print("PASS", firewall)

    mapping_path = ROOT / "audits" / "ARTS_EVALUATION_MAPPING_FINAL.json"
    manifest_path = ROOT / "manifests" / "ARTS_EVALUATION_MANIFEST.json"
    report_path = ROOT / "audits" / "ARTS_EVALUATION_INTEGRITY_REPORT.md"
    atomic_write_json(mapping_path, mapping)
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(report_path, "\n".join(report_lines) + "\n")

    # Hash the final audit artifacts after writing them.
    final_artifacts = {
        "mapping": {"path": str(mapping_path), "sha256": sha256_file(mapping_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "integrity_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
    }

    say("36-RUN MINIMAL CAMPAIGN — FINAL PRE-FLIGHT")
    preflight = campaign_preflight()
    print(json.dumps(preflight, indent=2))

    completion = {
        "status": "PASS_READY_NOT_LAUNCHED",
        "arts_commit": head,
        "evaluation_artifacts": final_artifacts,
        "campaign_preflight": preflight,
        "instruction": "Do not launch if any later integrity check changes these hashes.",
    }
    completion_path = ROOT / "audits" / "STEP9_COMPLETION.json"
    atomic_write_json(completion_path, completion)

    say("STEP 9 STATUS")
    print("Official ARTS suffix/family mapping          : PASS")
    print("Official ARTS schema + source lineage        : PASS")
    print("Derivative metadata anomaly audit            : PASS (preserved, not rewritten)")
    print("Model-input contract (offsets unused)        : PASS")
    print("Published family counts                      : PASS")
    print("Local clean-test ID-aware one-to-one alignment: PASS")
    print("Protected-test sentence firewall             : PASS")
    print("Six protected evaluation JSONLs              : FROZEN")
    print("36-run campaign final structural preflight   : PASS")
    print("Scientific training                           : NOT LAUNCHED")
    print("Completion record                             :", completion_path)
    print()
    print("NEXT COMMAND ONLY AFTER REVIEWING THIS PASS:")
    print("bash", ROOT / "campaigns" / "minimal_decisive" / "minimal_commands.sh")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n❌ STEP 9 FAIL-CLOSED:", exc, file=sys.stderr)
        raise
