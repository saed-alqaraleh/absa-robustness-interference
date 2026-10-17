from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable
from .schemas import AspectExample, OpinionRecord, GeneratedExample

POLAR_FLIP = {"positive":"negative", "negative":"positive"}

def deterministic_choice(items: list[str], key: str) -> str:
    if not items:
        raise ValueError("empty deterministic_choice")
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return sorted(items)[h % len(items)]

def _replace_spans(text: str, edits: list[tuple[int,int,str]]) -> str:
    # Edits must not overlap.
    edits = sorted(edits, key=lambda x: x[0], reverse=True)
    last_start = len(text) + 1
    for s,e,r in edits:
        if not (0 <= s <= e <= len(text)):
            raise ValueError(f"bad edit {(s,e,r)} for len={len(text)}")
        if e > last_start:
            raise ValueError("overlapping edits")
        text = text[:s] + r + text[e:]
        last_start = s
    return text

_WORDNET_CHECKED = False


def _require_wordnet() -> None:
    """Release v2: stop instead of silently replacing antonyms by negation when WordNet is missing."""
    global _WORDNET_CHECKED
    if _WORDNET_CHECKED:
        return
    try:
        from nltk.corpus import wordnet as wn
        version = wn.get_version()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("WordNet 3.0 is required by the reversal operator: "
                           "python -c \"import nltk; nltk.download('wordnet')\"") from exc
    if version != "3.0":
        raise RuntimeError(f"WordNet 3.0 is required by the reversal operator, found {version}")
    _WORDNET_CHECKED = True


def _wordnet_antonyms(word: str) -> list[str]:
    _require_wordnet()
    try:
        from nltk.corpus import wordnet as wn
        out = set()
        for syn in wn.synsets(word):
            for lem in syn.lemmas():
                for ant in lem.antonyms():
                    name = ant.name().replace("_", " ")
                    if name.lower() != word.lower():
                        out.add(name)
        return sorted(out)
    except Exception:
        return []

def _preserve_case(src: str, tgt: str) -> str:
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt

def reverse_opinion_text(span_text: str, key: str, antonym_lexicon: dict[str,list[str]] | None = None) -> tuple[str,str]:
    """
    Deterministic ARTS-inspired reversal.
    Prefer an antonym. If none exists, use deterministic negation.
    Returns transformed text and operation provenance.
    """
    antonym_lexicon = antonym_lexicon or {}
    words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", span_text))
    # prefer content words from right to left; an explicit lexicon can override WordNet.
    for m in reversed(words):
        w = m.group(0)
        cands = antonym_lexicon.get(w.lower(), []) or _wordnet_antonyms(w.lower())
        if cands:
            ant = deterministic_choice(cands, key + "|" + w.lower())
            ant = _preserve_case(w, ant)
            out = span_text[:m.start()] + ant + span_text[m.end():]
            return out, f"antonym:{w}->{ant}"
    # Negation fallback. Keep the rule explicit rather than pretending to match legacy ARTS exactly.
    aux = re.match(r"(?i)^\s*(is|are|was|were|am|can|could|will|would|should|has|have|had|do|does|did)\b", span_text)
    if aux:
        p = aux.end()
        return span_text[:p] + " not" + span_text[p:], "negation_after_aux"
    if re.match(r"(?i)^\s*not\b", span_text):
        return re.sub(r"(?i)^\s*not\s+", "", span_text, count=1), "remove_leading_negation"
    return "not " + span_text, "leading_negation"

def _simple_conjunction_adjust(sentence: str, aspects: list[AspectExample], new_labels: dict[str,str]) -> str:
    """
    Conservative conjunction repair: only switches a literal 'and'/'but' between
    adjacent aspect spans when the relation implied by their labels changes.
    """
    ordered = sorted(aspects, key=lambda x: x.aspect_from)
    edits = []
    for a,b in zip(ordered, ordered[1:]):
        between = sentence[a.aspect_to:b.aspect_from]
        m = re.search(r"\b(and|but)\b", between, flags=re.I)
        if not m:
            continue
        la, lb = new_labels.get(a.source_instance_id,a.label), new_labels.get(b.source_instance_id,b.label)
        if "neutral" in {la,lb}:
            continue
        desired = "and" if la == lb else "but"
        current = m.group(1)
        if current.lower() != desired:
            s = a.aspect_to + m.start(1)
            e = a.aspect_to + m.end(1)
            repl = desired.capitalize() if current[:1].isupper() else desired
            edits.append((s,e,repl))
    if not edits:
        return sentence
    return _replace_spans(sentence, edits)

@dataclass(frozen=True)
class AspectExpression:
    expression_id: str
    text: str
    label: str
    source_instance_id: str
    cluster_id: str
    domain: str

def build_aspect_expression_library(
    examples: list[AspectExample],
    opinions: dict[str, OpinionRecord],
    max_tokens: int = 14
) -> list[AspectExpression]:
    """
    Train-only deterministic alternative to ARTS's constituency-branch AspectSet.
    It extracts the minimal surface span covering the aspect and the first linked
    opinion span. It is therefore explicitly ARTS-inspired, not official ARTS.
    """
    out = []
    seen = set()
    for ex in examples:
        if ex.label not in {"positive","negative"}:
            continue
        op = opinions.get(ex.source_instance_id)
        if not op or not op.spans:
            continue
        o = op.spans[0]
        s = min(ex.aspect_from, o.start)
        e = max(ex.aspect_to, o.end)
        text = ex.sentence[s:e].strip(" ,;:.!?")
        if not text or len(text.split()) > max_tokens:
            continue
        # Require the actual aspect surface form to be present.
        if ex.aspect_term.lower() not in text.lower():
            continue
        key = (ex.domain, ex.label, text.lower())
        if key in seen:
            continue
        seen.add(key)
        eid = hashlib.sha256(("|".join(map(str,key))).encode()).hexdigest()[:16]
        out.append(AspectExpression(eid, text, ex.label, ex.source_instance_id, ex.cluster_id, ex.domain))
    return out

def eligibility(
    target: AspectExample,
    sentence_aspects: list[AspectExample],
    opinions: dict[str, OpinionRecord],
    family: str,
    aspect_library: list[AspectExpression] | None = None
) -> tuple[bool,str]:
    if family == "RevTgt":
        if target.label not in POLAR_FLIP:
            return False, "neutral_target_excluded_until_legacy_behavior_verified"
        if target.source_instance_id not in opinions:
            return False, "missing_target_opinion_span"
        return True, "eligible"
    if family == "RevNon":
        if target.label not in POLAR_FLIP:
            return False, "neutral_target_excluded_until_legacy_behavior_verified"
        non = [x for x in sentence_aspects if x.source_instance_id != target.source_instance_id]
        if not non:
            return False, "single_aspect_sentence"
        modifiable = [x for x in non if x.label in POLAR_FLIP]
        if not modifiable:
            return False, "no_polar_non_target"
        if any(x.source_instance_id not in opinions for x in modifiable):
            return False, "missing_non_target_opinion_span"
        return True, "eligible"
    if family == "AddDiff":
        if target.label not in POLAR_FLIP:
            return False, "neutral_target_excluded_for_common_pool"
        libs = [x for x in (aspect_library or []) if x.domain == target.domain and x.label != target.label and x.cluster_id != target.cluster_id]
        if not libs:
            return False, "no_train_only_opposite_expression"
        return True, "eligible"
    raise ValueError(f"unknown family {family}")

def generate_revtgt(
    target: AspectExample,
    sentence_aspects: list[AspectExample],
    opinions: dict[str, OpinionRecord],
    seed: int,
    antonym_lexicon: dict[str,list[str]] | None = None,
    adjust_conjunctions: bool = True
) -> GeneratedExample:
    ok, reason = eligibility(target, sentence_aspects, opinions, "RevTgt")
    if not ok:
        raise ValueError(reason)
    op = opinions[target.source_instance_id].spans[0]
    repl, operation = reverse_opinion_text(op.text, f"{seed}|{target.source_instance_id}|RevTgt", antonym_lexicon)
    sent = _replace_spans(target.sentence, [(op.start, op.end, repl)])
    new_label = POLAR_FLIP[target.label]
    # Conservative conjunction repair is disabled if the opinion edit changes string length,
    # because original character offsets would no longer be valid.
    if adjust_conjunctions and len(repl) == (op.end-op.start):
        sent = _simple_conjunction_adjust(sent, sentence_aspects, {target.source_instance_id:new_label})
    return GeneratedExample(
        domain=target.domain, source_split=target.split,
        source_instance_id=target.source_instance_id, cluster_id=target.cluster_id,
        original_sentence=target.sentence, original_aspect=target.aspect_term,
        original_label=target.label, perturbation_family="RevTgt",
        transformed_sentence=sent, transformed_aspect=target.aspect_term,
        transformed_label=new_label, modified_aspects=(target.aspect_term,),
        generation_seed=seed, eligibility_reason=reason,
        provenance="ARTS-inspired deterministic RevTgt; TOWE opinion span; WordNet/negation",
        metadata={"opinion_span":[op.start,op.end], "operation":operation}
    )

def generate_revnon(
    target: AspectExample,
    sentence_aspects: list[AspectExample],
    opinions: dict[str, OpinionRecord],
    seed: int,
    antonym_lexicon: dict[str,list[str]] | None = None,
    degree_adverbs: tuple[str,...] = ("very","really","extremely")
) -> GeneratedExample:
    ok, reason = eligibility(target, sentence_aspects, opinions, "RevNon")
    if not ok:
        raise ValueError(reason)
    edits = []
    modified = []
    operations = []
    new_labels = {}
    for x in sentence_aspects:
        if x.source_instance_id == target.source_instance_id or x.label not in POLAR_FLIP:
            continue
        op = opinions[x.source_instance_id].spans[0]
        if x.label == target.label:
            repl, operation = reverse_opinion_text(op.text, f"{seed}|{target.source_instance_id}|{x.source_instance_id}|RevNon", antonym_lexicon)
            edits.append((op.start, op.end, repl))
            new_labels[x.source_instance_id] = POLAR_FLIP[x.label]
            operations.append({"aspect":x.aspect_term,"operation":operation})
        else:
            adv = deterministic_choice(list(degree_adverbs), f"{seed}|{target.source_instance_id}|{x.source_instance_id}|adv")
            repl = f"{adv} {op.text}"
            edits.append((op.start, op.end, repl))
            new_labels[x.source_instance_id] = x.label
            operations.append({"aspect":x.aspect_term,"operation":f"intensify:{adv}"})
        modified.append(x.aspect_term)
    sent = _replace_spans(target.sentence, edits)
    return GeneratedExample(
        domain=target.domain, source_split=target.split,
        source_instance_id=target.source_instance_id, cluster_id=target.cluster_id,
        original_sentence=target.sentence, original_aspect=target.aspect_term,
        original_label=target.label, perturbation_family="RevNon",
        transformed_sentence=sent, transformed_aspect=target.aspect_term,
        transformed_label=target.label, modified_aspects=tuple(modified),
        generation_seed=seed, eligibility_reason=reason,
        provenance="ARTS-inspired deterministic RevNon; TOWE opinion spans; WordNet/negation/intensifier",
        metadata={"operations":operations}
    )

def generate_adddiff(
    target: AspectExample,
    sentence_aspects: list[AspectExample],
    opinions: dict[str, OpinionRecord],
    aspect_library: list[AspectExpression],
    seed: int,
    k_min: int = 1,
    k_max: int = 3
) -> GeneratedExample:
    ok, reason = eligibility(target, sentence_aspects, opinions, "AddDiff", aspect_library)
    if not ok:
        raise ValueError(reason)
    existing = target.sentence.lower()
    candidates = [
        x for x in aspect_library
        if x.domain == target.domain and x.label != target.label
        and x.cluster_id != target.cluster_id and x.text.lower() not in existing
    ]
    key = f"{seed}|{target.source_instance_id}|AddDiff"
    k_span = max(1, k_max-k_min+1)
    k = k_min + (int(hashlib.sha256((key+"|k").encode()).hexdigest(),16) % k_span)
    # deterministic sample without replacement by hash ranking
    candidates = sorted(candidates, key=lambda x: hashlib.sha256((key+"|"+x.expression_id).encode()).hexdigest())
    chosen = candidates[:min(k,len(candidates))]
    if not chosen:
        raise ValueError("no_adddiff_candidate_after_overlap_filter")
    connector = ", but "
    suffix = connector + ", and ".join(x.text for x in chosen)
    sent = target.sentence.rstrip()
    if sent.endswith((".", "!", "?")):
        sent = sent[:-1]
    sent = sent + suffix + "."
    return GeneratedExample(
        domain=target.domain, source_split=target.split,
        source_instance_id=target.source_instance_id, cluster_id=target.cluster_id,
        original_sentence=target.sentence, original_aspect=target.aspect_term,
        original_label=target.label, perturbation_family="AddDiff",
        transformed_sentence=sent, transformed_aspect=target.aspect_term,
        transformed_label=target.label, modified_aspects=tuple(x.text for x in chosen),
        generation_seed=seed, eligibility_reason=reason,
        provenance="ARTS-inspired train-only AddDiff using deterministic minimal-cover aspect-expression library",
        metadata={"expression_ids":[x.expression_id for x in chosen], "k":len(chosen)}
    )
