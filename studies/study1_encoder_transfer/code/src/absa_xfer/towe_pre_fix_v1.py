from __future__ import annotations
import ast, csv, re, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .schemas import AspectExample, OpinionRecord, OpinionSpan

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

def _parse_sequence_field(value: str) -> list[str]:
    v = value.strip()
    if not v:
        return []
    if v[0] in "[(":
        try:
            obj = ast.literal_eval(v)
            if isinstance(obj, (list, tuple)):
                return [str(x) for x in obj]
        except Exception:
            pass
    # TOWE derivatives in the wild use spaces, backslashes, or commas.
    for sep in ("\t", "\\", ","):
        if sep in v and " " not in v:
            return [x for x in (p.strip() for p in v.split(sep)) if x]
    return v.split()

def _spans_from_bio(tags: list[str]) -> list[tuple[int,int]]:
    spans = []
    start = None
    for i, t in enumerate(tags + ["O"]):
        u = t.upper().strip()
        is_b = u.startswith("B") or u in {"1", "S"}
        is_i = u.startswith("I") or u in {"2", "E"}
        if start is None and (is_b or is_i):
            start = i
        elif start is not None and is_b:
            spans.append((start, i))
            start = i
        elif start is not None and not is_i:
            spans.append((start, i))
            start = None
    return spans

def _token_offsets(sentence: str, tokens: list[str]) -> list[tuple[int,int]] | None:
    offsets = []
    pos = 0
    lower = sentence.lower()
    for tok in tokens:
        t = tok
        if t in {"-LRB-", "-RRB-"}:
            t = {"-LRB-":"(", "-RRB-":")"}[t]
        # tolerate tokenizers that separate punctuation and normalize quotes
        candidates = [t, t.replace("``", '"').replace("''", '"')]
        idx = -1
        used = None
        for c in candidates:
            idx = lower.find(c.lower(), pos)
            if idx >= 0:
                used = c
                break
        if idx < 0:
            return None
        offsets.append((idx, idx + len(used)))
        pos = idx + len(used)
    return offsets

@dataclass
class TOWERow:
    sentence: str
    tokens: list[str]
    target_tags: list[str]
    opinion_tags: list[str]
    row_id: str

def load_towe_tsv(path: str | Path) -> list[TOWERow]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sentence", "target_tags", "opinion_words_tags"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"TOWE TSV missing columns {required}: {path}")
        for i, r in enumerate(reader):
            tokens = _parse_sequence_field(r["sentence"])
            tt = _parse_sequence_field(r["target_tags"])
            ot = _parse_sequence_field(r["opinion_words_tags"])
            if not (len(tokens) == len(tt) == len(ot)):
                raise ValueError(f"Length mismatch at {path}:{i+2}")
            rows.append(TOWERow(
                sentence=" ".join(tokens), tokens=tokens, target_tags=tt,
                opinion_tags=ot, row_id=r.get("s_id", str(i))
            ))
    return rows

def align_towe_to_examples(examples: list[AspectExample], towe_rows: list[TOWERow], provenance: str) -> tuple[dict[str, OpinionRecord], list[dict]]:
    """
    Fully automatic alignment. Ambiguous/unmatched items are reported and not retained.
    No manual retain/drop decisions are allowed.
    """
    by_norm = {}
    for r in towe_rows:
        by_norm.setdefault(normalize_text(r.sentence), []).append(r)

    aligned = {}
    report = []
    for ex in examples:
        candidates = by_norm.get(normalize_text(ex.sentence), [])
        # fallback: punctuation-normalized token rendering
        if not candidates:
            n = normalize_text(re.sub(r"\s+([,.;:!?])", r"\1", ex.sentence))
            candidates = by_norm.get(n, [])
        matches = []
        for r in candidates:
            target_spans = _spans_from_bio(r.target_tags)
            offs = _token_offsets(ex.sentence, r.tokens)
            if offs is None:
                continue
            for ts, te in target_spans:
                if ts >= len(offs) or te <= ts:
                    continue
                cs, ce = offs[ts][0], offs[te-1][1]
                if normalize_text(ex.sentence[cs:ce]) == normalize_text(ex.aspect_term):
                    op_spans = []
                    for os_, oe in _spans_from_bio(r.opinion_tags):
                        if os_ >= len(offs) or oe <= os_:
                            continue
                        s, e = offs[os_][0], offs[oe-1][1]
                        op_spans.append(OpinionSpan(s, e, ex.sentence[s:e], provenance))
                    if op_spans:
                        matches.append((r, tuple(op_spans)))
        if len(matches) == 1:
            r, spans = matches[0]
            aligned[ex.source_instance_id] = OpinionRecord(
                source_instance_id=ex.source_instance_id,
                cluster_id=ex.cluster_id,
                spans=spans, matched_by="normalized_sentence+target_span", provenance=provenance
            )
            report.append({"source_instance_id": ex.source_instance_id, "status":"matched", "towe_row":r.row_id})
        else:
            report.append({
                "source_instance_id": ex.source_instance_id,
                "status":"ambiguous" if len(matches)>1 else "unmatched",
                "num_matches": len(matches)
            })
    return aligned, report
