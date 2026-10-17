from __future__ import annotations
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from .schemas import AspectExample, VALID_LABELS
from .hashing import short_hash

def _ids(domain: str, sentence_id: str, start: int, end: int, term: str):
    cluster_id = short_hash({"domain": domain, "sentence_id": sentence_id})
    source_instance_id = short_hash({
        "domain": domain, "sentence_id": sentence_id,
        "start": start, "end": end, "term": term
    })
    return source_instance_id, cluster_id

def parse_semeval_xml(path: str | Path, domain: str, split: str, drop_conflict: bool = True) -> list[AspectExample]:
    tree = ET.parse(path)
    root = tree.getroot()
    out: list[AspectExample] = []
    for sent in root.findall(".//sentence"):
        sid = sent.attrib.get("id", "")
        text_el = sent.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text
        terms = sent.find("aspectTerms")
        if terms is None:
            continue
        for term in terms.findall("aspectTerm"):
            label = term.attrib.get("polarity", "").strip().lower()
            if label == "conflict" and drop_conflict:
                continue
            if label not in VALID_LABELS:
                continue
            aspect = term.attrib.get("term", "")
            start = int(term.attrib.get("from", "-1"))
            end = int(term.attrib.get("to", "-1"))
            if start < 0 or end < start or text[start:end] != aspect:
                raise ValueError(
                    f"Invalid aspect span in {path}: sentence={sid!r}, "
                    f"term={aspect!r}, span=({start},{end}), slice={text[start:end]!r}"
                )
            source_instance_id, cluster_id = _ids(domain, sid, start, end, aspect)
            out.append(AspectExample(
                domain=domain, split=split, sentence_id=sid, sentence=text,
                aspect_term=aspect, aspect_from=start, aspect_to=end, label=label,
                source_instance_id=source_instance_id, cluster_id=cluster_id
            ))
    return out

def group_by_sentence(examples: list[AspectExample]) -> dict[str, list[AspectExample]]:
    groups = defaultdict(list)
    for x in examples:
        groups[x.cluster_id].append(x)
    return dict(groups)
