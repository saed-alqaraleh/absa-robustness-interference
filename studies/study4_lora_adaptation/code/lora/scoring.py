"""Prompt rendering and label scoring, identical in rule to Study III (llm_kit_v1/llm/run_llm.py).

score(label) = logsumexp over the first tokens of the label's surface forms ("positive",
"Positive", " positive", " Positive", ...) of the next-token log-probability after the chat-template
assistant prefix (thinking disabled). Tokens shared between labels are dropped. The prediction is the
argmax of the three scores. The same function supplies the training objective (cross-entropy over the
three scores), so training and evaluation use one decision rule.
"""

from __future__ import annotations

import inspect

from lora.common import LABELS, VARIANTS, log, user_message


def label_token_ids(tok) -> dict[str, list[int]]:
    first = {lab: {tok.encode(v, add_special_tokens=False)[0] for v in VARIANTS[lab]} for lab in LABELS}
    shared = {t for a in LABELS for b in LABELS if a < b for t in first[a] & first[b]}
    ids = {lab: sorted(first[lab] - shared) for lab in LABELS}
    empty = [lab for lab in LABELS if not ids[lab]]
    if empty:
        raise SystemExit(f"no unambiguous first token for {empty} with this tokenizer: {first}")
    return ids


def render(tok, content: str) -> str:
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(tok, d: str, items: list[dict], max_tokens: int) -> list[list[int]]:
    texts = [render(tok, user_message(d, it)) for it in items]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    too_long = [i for i, x in enumerate(ids) if len(x) > max_tokens]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts exceed {max_tokens} tokens (first index {too_long[0]})")
    return ids


def pad_id(tok) -> int:
    if tok.pad_token_id is not None:
        return tok.pad_token_id
    return tok.eos_token_id if tok.eos_token_id is not None else 0


def left_pad(seqs: list[list[int]], pad: int, device):
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, L - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[r, L - len(s):] = 1
    return ids.to(device), mask.to(device)


def _keeps_last_only(model) -> dict:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(base.forward).parameters else {}


def label_scores(model, ids, mask, lab_index):
    """[batch, 3] label log-probabilities at the last prompt position (differentiable)."""
    import torch
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, **_keeps_last_only(model))
    lp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
    return torch.stack([torch.logsumexp(lp[:, ix], dim=-1) for ix in lab_index], dim=-1)


def label_index(label_ids: dict[str, list[int]], device):
    import torch
    return [torch.tensor(label_ids[lab], device=device) for lab in LABELS]


def score_items(model, tok, enc: list[list[int]], label_ids, batch_size: int, device) -> tuple[list[list[float]], int]:
    """Score pre-encoded prompts; halves the batch on CUDA out-of-memory. Returns (scores, batch size used)."""
    import torch
    lab_index = label_index(label_ids, device)
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))       # longest first: OOM shows at once
    B = batch_size
    while True:
        try:
            scores: list = [None] * len(enc)
            was_training = model.training
            model.eval()
            with torch.inference_mode():
                for s in range(0, len(order), B):
                    chunk = order[s:s + B]
                    ids, mask = left_pad([enc[i] for i in chunk], pad_id(tok), device)
                    out = label_scores(model, ids, mask, lab_index).cpu().tolist()
                    for r, i in enumerate(chunk):
                        scores[i] = out[r]
            if was_training:
                model.train()
            return scores, B
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or B == 1:
                raise
            torch.cuda.empty_cache()
            B = max(1, B // 2)
            log("score", f"CUDA out of memory: retrying with batch size {B}")
