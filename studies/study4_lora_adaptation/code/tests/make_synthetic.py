#!/usr/bin/env python3
"""Build a synthetic Study I root (full frozen sizes) and a tiny Qwen3 model + word-level tokenizer with a
Qwen3-style chat template, for CPU tests of the Study IV kit. Nothing here touches real data.

  python tests/make_synthetic.py OUT_DIR      -> OUT_DIR/root, OUT_DIR/tiny_qwen3
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lora.common import CONDITIONS, EXPECTED_EVAL, EXPECTED_TRAIN_ROWS, INSTRUCTION, LABELS, VARIANTS, domain_word  # noqa: E402

ASPECTS = {"restaurant": ["food", "service", "staff", "pasta", "wine", "price", "ambience", "dessert"],
           "laptop": ["battery", "screen", "keyboard", "price", "speakers", "trackpad", "fan", "charger"]}
CUE = {"positive": ["great", "excellent", "lovely"], "negative": ["awful", "terrible", "bad"],
       "neutral": ["ordinary", "usual", "standard"]}
B = {"restaurant": 1004, "laptop": 419}

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


def sentence(rng, aspects_labels):
    parts = [f"the {a} was {rng.choice(CUE[l])}" for a, l in aspects_labels]
    return " and ".join(parts) + " ."


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_root(root: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    for d in ("restaurant", "laptop"):
        A = ASPECTS[d]

        def item(n_asp=1):
            asp = rng.sample(A, n_asp)
            labs = [rng.choice(LABELS) for _ in asp]
            return asp, labs

        clean_base = []
        for _ in range(EXPECTED_TRAIN_ROWS[d] - B[d]):
            asp, labs = item(rng.choice([1, 2]))
            s = sentence(rng, list(zip(asp, labs)))
            clean_base.append({"sentence": s, "aspect_term": asp[0], "label": labs[0], "is_augmented": False})
        for cond in ("Clean-Natural",) + CONDITIONS:
            extra = []
            for _ in range(B[d]):
                asp, labs = item(2)
                if cond in ("Clean+RevTgt", "Clean+Mixed") and rng.random() < 0.5:   # flip target polarity
                    labs[0] = {"positive": "negative", "negative": "positive", "neutral": "neutral"}[labs[0]]
                extra.append({"sentence": sentence(rng, list(zip(asp, labs))), "aspect_term": asp[0], "label": labs[0],
                              "is_augmented": cond != "Clean-Matched",
                              "generation": {"metadata": {"operations": [cond]}}})
            rows = clean_base + (extra if cond != "Clean-Natural" else [])
            write_jsonl(root / "transformations" / "materialized" / d / cond / "train.jsonl", rows)
        dev = []
        for k in range(150):
            asp, labs = item(rng.choice([1, 2]))
            dev.append({"sentence": sentence(rng, list(zip(asp, labs))), "aspect_term": asp[0], "label": labs[0],
                        "source_instance_id": f"{d}-dev-{k}", "cluster_id": f"{d}-devsent-{k}"})
        write_jsonl(root / "data_processed" / d / "dev.jsonl", dev)
        write_jsonl(root / "data_processed" / d / "train.jsonl", clean_base)
        # test: sentences with 1-2 aspects; each (sentence, aspect) is a source instance
        n = EXPECTED_EVAL[d]
        test, sk = [], 0
        while len(test) < n["Clean"]:
            k = min(rng.choice([1, 2]), n["Clean"] - len(test))
            asp, labs = item(k)
            s = sentence(rng, list(zip(asp, labs)))
            for a, l in zip(asp, labs):
                test.append({"sentence": s, "aspect_term": a, "label": l,
                             "source_instance_id": f"{d}-src-{len(test)}", "cluster_id": f"{d}-sent-{sk}"})
            sk += 1
        cluster_missing = 3                      # a few records without cluster_id exercise the fallback
        write_jsonl(root / "data_processed" / d / "test.jsonl", test)
        flip = {"positive": "negative", "negative": "positive", "neutral": "positive"}
        rt_src = test[:n["RevTgt"]]
        rn_src = rt_src[:n["RevNon"]]
        fam = {"RevTgt": [dict(t, sentence=t["sentence"].replace("was", "was not", 1), label=flip[t["label"]])
                          for t in rt_src],
               "RevNon": [dict(t, sentence=t["sentence"] + " but the rest was bad .") for t in rn_src],
               "AddDiff": [dict(t, sentence=t["sentence"] + " the music was great .") for t in test]}
        for f, rows in fam.items():
            if f == "AddDiff":
                for r in rows[-cluster_missing:]:
                    r.pop("cluster_id")
            write_jsonl(root / "data_processed" / d / "evaluation" / f"{f}.jsonl", rows)


def build_model(out: Path, root: Path) -> None:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    words = set()
    texts = [INSTRUCTION, domain_word("restaurant"), domain_word("laptop"), "The sentences are",
             "Sentence: Aspect: Sentiment:", "user assistant system"]
    for p in root.rglob("*.jsonl"):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            texts += [r["sentence"], r["aspect_term"]]
    pre = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    for t in texts:
        words.update(w for w, _ in pre.pre_tokenize_str(t))
    for lab in LABELS:
        words.update(v.strip() for v in VARIANTS[lab])
    special = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"]
    vocab = {w: i for i, w in enumerate(special + sorted(words - set(special)))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  eos_token="<|im_end|>", additional_special_tokens=special[2:])
    tok.chat_template = CHAT_TEMPLATE
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True, pad_token_id=vocab["<pad>"], eos_token_id=vocab["<|im_end|>"])
    Qwen3ForCausalLM(cfg).save_pretrained(out)
    tok.save_pretrained(out)


if __name__ == "__main__":
    base = Path(sys.argv[1]).resolve()
    build_root(base / "root")
    build_model(base / "tiny_qwen3", base / "root")
    print(f"synthetic root and tiny model written under {base}")
