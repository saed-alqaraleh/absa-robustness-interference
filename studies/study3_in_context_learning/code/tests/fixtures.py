"""Synthetic frozen project root and a tiny local RoBERTa for offline tests (no network, no GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

WORDS = ("the food was good bad great terrible service staff nice rude price cheap expensive not very really "
         "extremely and but place menu wine decor ok fine slow fast tasty bland").split()


def _id(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def make_root(tmp: Path, n_sentences: int = 20) -> Path:
    """Two aspects per sentence (food, service), polar labels, frozen derivatives for every source."""
    root = tmp / "root"
    for d in ("restaurant", "laptop"):
        train, fam = [], {"RT": [], "RN": [], "AD": []}
        for k in range(n_sentences):
            lab_f = "positive" if k % 2 == 0 else "negative"
            lab_s = lab_f if k % 3 else ("negative" if lab_f == "positive" else "positive")
            op_f = "good" if lab_f == "positive" else "bad"
            op_s = "nice" if lab_s == "positive" else "rude"
            sent = f"the food was {op_f} and the service was {op_s}"
            cid = _id(d, "c", k)
            for asp, lab, op in (("food", lab_f, op_f), ("service", lab_s, op_s)):
                sid = _id(d, k, asp)
                row = {"aspect_from": sent.index(asp), "aspect_term": asp, "aspect_to": sent.index(asp) + len(asp),
                       "cluster_id": cid, "domain": d, "label": lab, "sentence": sent, "sentence_id": str(k),
                       "source_instance_id": sid, "split": "train"}
                train.append(row)
                s = sent.index(op, sent.index(asp))
                e = s + len(op)
                flip = "negative" if lab == "positive" else "positive"
                ant = {"good": "bad", "bad": "good", "nice": "rude", "rude": "nice"}[op]
                gen = {"original_sentence": sent, "original_label": lab, "source_instance_id": sid}
                fam["RT"].append({**row, "is_augmented": True, "perturbation_family": "RevTgt", "label": flip,
                                  "sentence": sent[:s] + ant + sent[e:],
                                  "generation": {**gen, "metadata": {"opinion_span": [s, e], "operation": f"antonym:{op}->{ant}"}}})
                other_op = "intensify:very" if (k % 5 == 0) else "antonym:x->y"
                fam["RN"].append({**row, "is_augmented": True, "perturbation_family": "RevNon",
                                  "sentence": sent + " (non-target edited)",
                                  "generation": {**gen, "metadata": {"operations": [{"aspect": "x", "operation": other_op}]}}})
                fam["AD"].append({**row, "is_augmented": True, "perturbation_family": "AddDiff",
                                  "sentence": sent + " but the decor was bland", "generation": {**gen, "metadata": {}}})
        dp = root / "data_processed" / d
        write_jsonl(dp / "train.jsonl", train)
        dev = [{**r, "split": "dev", "source_instance_id": _id("dev", r["source_instance_id"])} for r in train[:12]]
        test = [{**r, "split": "test", "source_instance_id": _id("test", r["source_instance_id"]),
                 "cluster_id": _id("tc", r["cluster_id"])} for r in train[:16]]
        write_jsonl(dp / "dev.jsonl", dev)
        write_jsonl(dp / "test.jsonl", test)
        flipd = {"positive": "negative", "negative": "positive"}
        write_jsonl(dp / "evaluation" / "RevTgt.jsonl", [{**r, "label": flipd[r["label"]], "sentence": r["sentence"] + " !"} for r in test])
        write_jsonl(dp / "evaluation" / "RevNon.jsonl", [{**r, "sentence": r["sentence"] + " ?"} for r in test[:8]])
        write_jsonl(dp / "evaluation" / "AddDiff.jsonl", [{**r, "sentence": r["sentence"] + " and ok"} for r in test])
        for f, cond in (("RT", "Clean+RevTgt"), ("RN", "Clean+RevNon"), ("AD", "Clean+AddDiff")):
            clean = [{**r, "is_augmented": False, "perturbation_family": "Clean"} for r in train]
            write_jsonl(root / "transformations" / "materialized" / d / cond / "train.jsonl", clean + fam[f])
        mixed = [{**r, "is_augmented": False, "perturbation_family": "Clean"} for r in train] + fam["RT"][:4] + fam["RN"][:4] + fam["AD"][:4]
        write_jsonl(root / "transformations" / "materialized" / d / "Clean+Mixed" / "train.jsonl", mixed)
        sel = root / "audits" / "eligibility" / d / "FINAL_GENERATION_VALID_SOURCE_SELECTION.json"
        sel.parent.mkdir(parents=True, exist_ok=True)
        sel.write_text(json.dumps({"selected_ids": [r["source_instance_id"] for r in train]}))
        rd = root / "stage1_v1" / "data" / d / "robust_dev"
        devrt = [{**r, "label": flipd[r["label"]], "orig_label": r["label"], "stage1_family": "RT"} for r in dev[:8]]
        write_jsonl(rd / "RT.jsonl", devrt)
        write_jsonl(rd / "RN.jsonl", [{**r, "orig_label": r["label"], "stage1_family": "RN"} for r in dev[:6]])
        write_jsonl(rd / "AD.jsonl", [{**r, "orig_label": r["label"], "stage1_family": "AD"} for r in dev[:8]])
    return root


def make_tiny_causal_lm(tmp: Path, root: Path) -> Path:
    """Tiny random Llama with a byte-level BPE tokenizer trained on the fixture text and a chat template."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from llm.common import INSTRUCTION
    out = tmp / "tiny_llama"
    corpus = []
    for p in root.rglob("*.jsonl"):
        corpus += [json.loads(x)["sentence"] for x in p.read_text().splitlines() if x.strip()]
    corpus += [INSTRUCTION, "The sentences are restaurant reviews.", "The sentences are laptop reviews.",
               "Examples:", "Now classify this one.", "Sentence: Aspect: Sentiment:"]
    corpus += ["positive Positive negative Negative neutral Neutral"] * 50
    specials = ["<s>", "</s>", "<pad>", "<|user|>", "<|end|>", "<|assistant|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus * 3, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tok.chat_template = ("<s>{% for m in messages %}<|user|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
    tok.save_pretrained(out)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(out)
    return out
