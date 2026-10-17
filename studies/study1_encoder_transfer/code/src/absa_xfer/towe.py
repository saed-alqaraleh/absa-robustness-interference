
from __future__ import annotations

import csv
import re
import unicodedata

from dataclasses import dataclass
from pathlib import Path

from .schemas import (
    AspectExample,
    OpinionRecord,
    OpinionSpan,
)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_text(s: str) -> str:
    """
    Conservative ordinary normalization.

    Used for surface-form comparisons, not for changing model input.
    """
    s = unicodedata.normalize(
        "NFKC",
        s or "",
    )

    s = s.casefold()

    s = re.sub(
        r"\s+",
        " ",
        s,
    ).strip()

    return s


def normalize_compact(s: str) -> str:
    """
    Tokenization-insensitive normalization.

    Examples:
        "food ,"    -> "food,"
        "food,"     -> "food,"
        "could n't" -> "couldn't"
        "couldn't"  -> "couldn't"

    Only whitespace is removed.
    Punctuation and lexical content are preserved.
    """
    s = unicodedata.normalize(
        "NFKC",
        s or "",
    )

    s = s.casefold()

    return re.sub(
        r"\s+",
        "",
        s,
    )


def normalize_alnum(s: str) -> str:
    """
    Conservative fallback key.

    This is used only for candidate retrieval when compact matching fails.

    A fallback is accepted only if it identifies exactly one
    tokenized sentence surface before target matching.
    """
    s = unicodedata.normalize(
        "NFKC",
        s or "",
    ).casefold()

    return "".join(
        ch
        for ch in s
        if ch.isalnum()
    )


# =============================================================================
# TOWE TOKEN\TAG PARSING
# =============================================================================

_VALID_TAG_PREFIXES = {
    "B",
    "I",
    "O",
    "S",
    "E",
}


def _parse_annotated_field(
    value: str,
) -> tuple[list[str], list[str]]:
    """
    Parse the actual TOWE representation.

    TOWE fields look like:

        The\O food\B is\O good\O .\O

    They are NOT a simple sequence of BIO labels.
    Every whitespace-delimited item contains:

        token\BIO-tag
    """

    value = (value or "").strip()

    if not value:
        return [], []

    pieces = value.split()

    tokens = []
    tags = []

    for piece in pieces:

        if "\\" not in piece:
            raise ValueError(
                "Malformed TOWE token/tag item: "
                + repr(piece)
            )

        token, tag = piece.rsplit(
            "\\",
            1,
        )

        tag = tag.strip().upper()

        if not token:
            raise ValueError(
                "Empty token in TOWE field: "
                + repr(piece)
            )

        if (
            not tag
            or tag[0]
            not in _VALID_TAG_PREFIXES
        ):
            raise ValueError(
                "Invalid TOWE BIO tag: "
                + repr(tag)
            )

        tokens.append(token)
        tags.append(tag)

    return tokens, tags


# =============================================================================
# BIO SPANS
# =============================================================================

def _spans_from_bio(
    tags: list[str],
) -> list[tuple[int, int]]:

    spans = []
    start = None

    for i, raw_tag in enumerate(
        tags + ["O"]
    ):

        tag = (
            raw_tag
            .upper()
            .strip()
        )

        prefix = (
            tag[0]
            if tag
            else "O"
        )

        is_b = prefix in {
            "B",
            "S",
        }

        is_i = prefix in {
            "I",
            "E",
        }

        if start is None:

            if is_b or is_i:
                start = i

        else:

            if is_b:
                spans.append(
                    (start, i)
                )
                start = i

            elif not is_i:
                spans.append(
                    (start, i)
                )
                start = None

    return spans


# =============================================================================
# TOKEN -> ORIGINAL SEMEVAL CHARACTER OFFSETS
# =============================================================================

def _token_aliases(
    token: str,
) -> list[str]:

    mapping = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
        "``": '"',
        "''": '"',
    }

    primary = mapping.get(
        token,
        token,
    )

    variants = [
        primary,
        primary.replace(
            "’",
            "'",
        ),
        primary.replace(
            "‘",
            "'",
        ),
        primary.replace(
            "“",
            '"',
        ).replace(
            "”",
            '"',
        ),
    ]

    out = []

    for x in variants:
        if x not in out:
            out.append(x)

    return out


def _token_offsets(
    sentence: str,
    tokens: list[str],
) -> list[tuple[int, int]] | None:
    """
    Sequentially map TOWE tokens onto the original SemEval sentence.

    This naturally handles:
        food ,    -> food,
        could n't -> couldn't
        ' perks ' -> 'perks'

    because whitespace is not required between successive tokens.
    """

    offsets = []

    pos = 0
    lower = sentence.lower()

    for token in tokens:

        found = None

        for candidate in _token_aliases(
            token
        ):

            idx = lower.find(
                candidate.lower(),
                pos,
            )

            if idx >= 0:

                found = (
                    idx,
                    idx + len(candidate),
                )

                break

        if found is None:
            return None

        offsets.append(found)
        pos = found[1]

    return offsets


# =============================================================================
# TOWE ROW
# =============================================================================

@dataclass
class TOWERow:

    sentence: str

    tokens: list[str]

    target_tags: list[str]

    opinion_tags: list[str]

    row_id: str


# =============================================================================
# LOAD TOWE
# =============================================================================

def load_towe_tsv(
    path: str | Path,
) -> list[TOWERow]:

    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        required = {
            "sentence",
            "target_tags",
            "opinion_words_tags",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                f"TOWE TSV missing columns "
                f"{required}: {path}"
            )

        for i, r in enumerate(
            reader,
            start=2,
        ):

            sentence = (
                r["sentence"]
                or ""
            ).strip()

            sentence_tokens = (
                sentence.split()
            )

            (
                target_tokens,
                target_tags,
            ) = _parse_annotated_field(
                r["target_tags"]
            )

            (
                opinion_tokens,
                opinion_tags,
            ) = _parse_annotated_field(
                r["opinion_words_tags"]
            )

            # --------------------------------------------------
            # Structural integrity
            # --------------------------------------------------

            if not (
                len(sentence_tokens)
                ==
                len(target_tokens)
                ==
                len(target_tags)
                ==
                len(opinion_tokens)
                ==
                len(opinion_tags)
            ):

                raise ValueError(
                    f"TOWE length mismatch "
                    f"at {path}:{i}; "
                    f"sentence={len(sentence_tokens)}, "
                    f"target={len(target_tokens)}, "
                    f"opinion={len(opinion_tokens)}"
                )

            # The annotated fields repeat the actual tokens.
            # Verify that they represent the same sequence.
            if (
                normalize_compact(
                    " ".join(
                        target_tokens
                    )
                )
                !=
                normalize_compact(
                    sentence
                )
            ):

                raise ValueError(
                    f"Target token sequence differs "
                    f"from sentence at {path}:{i}"
                )

            if (
                normalize_compact(
                    " ".join(
                        opinion_tokens
                    )
                )
                !=
                normalize_compact(
                    sentence
                )
            ):

                raise ValueError(
                    f"Opinion token sequence differs "
                    f"from sentence at {path}:{i}"
                )

            rows.append(
                TOWERow(
                    sentence=sentence,
                    tokens=target_tokens,
                    target_tags=target_tags,
                    opinion_tags=opinion_tags,
                    row_id=r.get(
                        "s_id",
                        str(i - 2),
                    ),
                )
            )

    return rows


# =============================================================================
# ALIGNMENT
# =============================================================================

def align_towe_to_examples(
    examples: list[AspectExample],
    towe_rows: list[TOWERow],
    provenance: str,
) -> tuple[
    dict[str, OpinionRecord],
    list[dict],
]:

    """
    Fully automatic alignment.

    Matching stages:

      1. whitespace-insensitive sentence identity;
      2. conservative unique alphanumeric fallback;
      3. target BIO span must map to the actual SemEval aspect;
      4. at least one opinion BIO span must map to valid
         SemEval character offsets.

    Ambiguous or unmatched examples are never manually repaired.
    """

    # ----------------------------------------------------------
    # Candidate sentence indexes
    # ----------------------------------------------------------

    by_compact = {}
    by_alnum = {}

    for row in towe_rows:

        compact_key = normalize_compact(
            row.sentence
        )

        alnum_key = normalize_alnum(
            row.sentence
        )

        by_compact.setdefault(
            compact_key,
            [],
        ).append(row)

        by_alnum.setdefault(
            alnum_key,
            [],
        ).append(row)

    aligned = {}
    report = []

    for ex in examples:

        # ------------------------------------------------------
        # Stage 1 — compact sentence identity
        # ------------------------------------------------------

        compact_key = normalize_compact(
            ex.sentence
        )

        candidates = list(
            by_compact.get(
                compact_key,
                [],
            )
        )

        sentence_method = (
            "compact_sentence"
        )

        # ------------------------------------------------------
        # Stage 2 — conservative alphanumeric fallback
        # ------------------------------------------------------

        if not candidates:

            fallback = list(
                by_alnum.get(
                    normalize_alnum(
                        ex.sentence
                    ),
                    [],
                )
            )

            # There can legitimately be multiple TOWE rows
            # for different aspects of one sentence.
            #
            # The fallback is safe only if every retrieved row
            # represents the SAME compact tokenized sentence.
            compact_surfaces = {
                normalize_compact(
                    r.sentence
                )
                for r in fallback
            }

            if (
                fallback
                and
                len(compact_surfaces) == 1
            ):
                candidates = fallback
                sentence_method = (
                    "unique_alnum_sentence_fallback"
                )

            elif fallback:

                report.append({
                    "source_instance_id":
                        ex.source_instance_id,

                    "status":
                        "ambiguous",

                    "reason":
                        "alnum_sentence_collision",

                    "num_sentence_candidates":
                        len(fallback),
                })

                continue

        if not candidates:

            report.append({
                "source_instance_id":
                    ex.source_instance_id,

                "status":
                    "unmatched",

                "reason":
                    "no_sentence_candidate",

                "num_matches":
                    0,
            })

            continue

        # ------------------------------------------------------
        # Target-specific candidate matching
        # ------------------------------------------------------

        matches = []

        offset_failures = 0
        target_failures = 0
        no_opinion_failures = 0

        for row in candidates:

            offsets = _token_offsets(
                ex.sentence,
                row.tokens,
            )

            if offsets is None:
                offset_failures += 1
                continue

            target_spans = (
                _spans_from_bio(
                    row.target_tags
                )
            )

            if not target_spans:
                target_failures += 1
                continue

            row_target_matched = False

            for ts, te in target_spans:

                if (
                    ts < 0
                    or te <= ts
                    or te > len(offsets)
                ):
                    continue

                cs = offsets[ts][0]
                ce = offsets[te - 1][1]

                target_surface = (
                    ex.sentence[
                        cs:ce
                    ]
                )

                if (
                    normalize_compact(
                        target_surface
                    )
                    !=
                    normalize_compact(
                        ex.aspect_term
                    )
                ):
                    continue

                row_target_matched = True

                opinion_spans = []

                seen_opinion_offsets = set()

                for os_, oe in _spans_from_bio(
                    row.opinion_tags
                ):

                    if (
                        os_ < 0
                        or oe <= os_
                        or oe > len(offsets)
                    ):
                        continue

                    start = offsets[os_][0]
                    end = offsets[oe - 1][1]

                    key = (
                        start,
                        end,
                    )

                    if key in seen_opinion_offsets:
                        continue

                    seen_opinion_offsets.add(
                        key
                    )

                    text = ex.sentence[
                        start:end
                    ]

                    if not text:
                        continue

                    opinion_spans.append(
                        OpinionSpan(
                            start=start,
                            end=end,
                            text=text,
                            provenance=provenance,
                        )
                    )

                if opinion_spans:

                    matches.append(
                        (
                            row,
                            tuple(
                                opinion_spans
                            ),
                            sentence_method,
                        )
                    )

                else:
                    no_opinion_failures += 1

            if not row_target_matched:
                target_failures += 1

        # ------------------------------------------------------
        # Resolve result
        # ------------------------------------------------------

        if len(matches) == 1:

            (
                row,
                spans,
                method,
            ) = matches[0]

            aligned[
                ex.source_instance_id
            ] = OpinionRecord(
                source_instance_id=
                    ex.source_instance_id,

                cluster_id=
                    ex.cluster_id,

                spans=
                    spans,

                matched_by=
                    method
                    + "+target_bio_surface",

                provenance=
                    provenance,
            )

            report.append({
                "source_instance_id":
                    ex.source_instance_id,

                "status":
                    "matched",

                "towe_row":
                    row.row_id,

                "matched_by":
                    method
                    + "+target_bio_surface",

                "num_opinion_spans":
                    len(spans),
            })

        elif len(matches) > 1:

            report.append({
                "source_instance_id":
                    ex.source_instance_id,

                "status":
                    "ambiguous",

                "reason":
                    "multiple_target_specific_matches",

                "num_matches":
                    len(matches),

                "candidate_towe_rows":
                    [
                        m[0].row_id
                        for m in matches
                    ],
            })

        else:

            report.append({
                "source_instance_id":
                    ex.source_instance_id,

                "status":
                    "unmatched",

                "reason":
                    (
                        "candidate_sentence_but_"
                        "no_valid_target_opinion_match"
                    ),

                "num_sentence_candidates":
                    len(candidates),

                "offset_failures":
                    offset_failures,

                "target_failures":
                    target_failures,

                "no_opinion_failures":
                    no_opinion_failures,
            })

    return aligned, report
