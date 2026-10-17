#!/usr/bin/env python3
"""Run the LLM probe for one shard of jobs (restartable; several shards may run in parallel).

Speed
-----
* Each job's instruction + demonstrations form a prompt prefix shared by every test item. The
  prefix is encoded once and its key/value cache is reused for every batch (expanded once to the
  batch size and cropped back after each batch), so only the short item suffix is computed.
* Labels are scored in one forward pass from the next-token distribution (no generation, no
  parsing): score(label) = logsumexp of the log-probabilities of the first tokens of
  "positive"/"Positive"/" positive"/" Positive" (etc.). Only the last position's logits are kept.
* Suffixes are left-padded inside the batch (explicit position ids keep RoPE positions exact),
  length-bucketed, and the batch size halves automatically on CUDA out-of-memory.
* Items whose tokenisation does not split cleanly at the prefix boundary fall back to a full
  forward pass; the count is recorded.

Sharding
--------
Jobs are numbered in a fixed order. ``--shard i/N`` keeps job k when k % N == i (use one value per
Colab notebook); ``--subshard j/M`` further keeps the shard's own jobs whose rank r satisfies
r % M == j (use one value per worker process inside a notebook). Every job writes its own CSV and
DONE marker, so a restart skips finished jobs; a DONE marker with a different signature is never
overwritten.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.common import (  # noqa: E402
    CONDITIONS, DEFAULT_MODELS, DEMO_SEEDS, DOMAINS, EVAL_SETS, INSTRUCTION, K, LABELS, OUT_SUBDIR, SCORING_VERSION,
    atomic_write_json, item_text, job_id, jobs, log, read_jsonl, select_jobs, sha256_file, slug, stable, user_message,
)

FAM_FILES = {"RT": "Clean+RevTgt", "RN": "Clean+RevNon", "AD": "Clean+AddDiff"}
VARIANTS = {"negative": ("negative", "Negative", " negative", " Negative"),
            "neutral": ("neutral", "Neutral", " neutral", " Neutral"),
            "positive": ("positive", "Positive", " positive", " Positive")}
SENTINEL = "<<<QUERY_SLOT_7f3a>>>"
_FORCE_DEEPCOPY = False          # tests only: exercise the path used for caches that cannot be cropped


# ----------------------------------------------------------------------------- data
def load_eval(root: Path, d: str, max_items: int | None = None) -> list[dict]:
    """Clean test split + the three ARTS families (evaluation only)."""
    items = []
    paths = {"Clean": root / "data_processed" / d / "test.jsonl",
             **{f: root / "data_processed" / d / "evaluation" / f"{f}.jsonl" for f in EVAL_SETS[1:]}}
    for name, p in paths.items():
        rows = read_jsonl(p)
        if max_items:
            rows = rows[:max_items]
        for r in rows:
            items.append({"set": name, "sentence": r["sentence"], "aspect_term": r["aspect_term"], "label": r["label"],
                          "source_instance_id": r["source_instance_id"], "cluster_id": r.get("cluster_id", "")})
    return items


def revnon_reversal_only(rec: dict) -> bool:
    ops = [o["operation"] for o in rec.get("generation", {}).get("metadata", {}).get("operations", [])]
    return bool(ops) and not any(o.startswith("intensify") for o in ops)


def build_demos(root: Path, d: str) -> dict[str, list[dict]]:
    """Deterministic, model-independent demonstration sets from frozen *training* material only.

    Pool: polar clean training sources that have frozen RT, AD and reversal-only RN derivatives.
    Within one condition and seed every demonstration comes from a different source, except
    Triplet, which shows the three derivatives of each of two sources next to each other.
    """
    clean = {r["source_instance_id"]: r for r in read_jsonl(root / "data_processed" / d / "train.jsonl")}
    fam = {}
    for f, cond in FAM_FILES.items():
        fam[f] = {}
        for r in read_jsonl(root / "transformations" / "materialized" / d / cond / "train.jsonl"):
            if r.get("is_augmented"):
                fam[f].setdefault(r["source_instance_id"], r)
    pool = sorted(s for s in clean if clean[s]["label"] in ("positive", "negative") and s in fam["RT"] and s in fam["AD"]
                  and s in fam["RN"] and revnon_reversal_only(fam["RN"][s]))

    def ex(r: dict, fam_name: str) -> dict:
        return {"sentence": r["sentence"], "aspect_term": r["aspect_term"], "label": r["label"],
                "source_instance_id": r["source_instance_id"], "family": fam_name}

    out = {}
    for seed in DEMO_SEEDS:
        for cond in CONDITIONS[1:]:
            rng = random.Random(stable(f"{d}|{cond}|{seed}"))
            order = pool[:]
            rng.shuffle(order)
            used: set[str] = set()

            def take(source_fam: str, label: str, n: int) -> list[dict]:
                got = []
                for s in order:
                    if s in used:
                        continue
                    r = clean[s] if source_fam == "Clean" else fam[source_fam][s]
                    if r["label"] == label:
                        got.append(ex(r, source_fam))
                        used.add(s)
                        if len(got) == n:
                            break
                return got

            if cond in ("Clean", "RT", "RN"):
                demos = take(cond, "positive", K // 2) + take(cond, "negative", K // 2)
                rng.shuffle(demos)
            elif cond == "Mixed":
                demos = []
                for f in ("RT", "RN", "AD"):
                    demos += take(f, "positive", 1) + take(f, "negative", 1)
                rng.shuffle(demos)
            else:  # Triplet: two sources (one per original polarity), each shown as RT, RN, AD adjacently
                groups = []
                for lab in ("positive", "negative"):
                    s = next(s for s in order if s not in used and clean[s]["label"] == lab)
                    used.add(s)
                    groups.append([ex(fam["RT"][s], "RT"), ex(fam["RN"][s], "RN"), ex(fam["AD"][s], "AD")])
                rng.shuffle(groups)
                demos = [x for g in groups for x in g]
            if len(demos) != K:
                raise SystemExit(f"{d}/{cond}/seed{seed}: only {len(demos)} demonstrations could be drawn (pool {len(pool)})")
            out[f"{cond}__seed{seed}"] = demos
    return out


# ----------------------------------------------------------------------------- tokenizer
def label_token_ids(tok) -> dict[str, list[int]]:
    """First token of each surface variant; tokens shared between labels are dropped."""
    first = {lab: {tok.encode(v, add_special_tokens=False)[0] for v in VARIANTS[lab]} for lab in LABELS}
    shared = {t for a in LABELS for b in LABELS if a < b for t in first[a] & first[b]}
    ids = {lab: sorted(first[lab] - shared) for lab in LABELS}
    empty = [lab for lab in LABELS if not ids[lab]]
    if empty:
        raise SystemExit(f"no unambiguous first token for {empty} with this tokenizer: {first}")
    return ids


def render_template(tok, content: str) -> str:
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def split_prompt(tok, d: str, demos: list[dict]) -> tuple[str, str]:
    rendered = render_template(tok, user_message(d, demos, None) + SENTINEL)
    if rendered.count(SENTINEL) != 1:
        raise SystemExit("chat template altered the query slot")
    pre, post = rendered.split(SENTINEL)
    return pre, post


# ----------------------------------------------------------------------------- scoring
def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def score_job(model, tok, d: str, demos: list[dict], items: list[dict], label_ids, batch_size: int,
              use_cache: bool, device) -> tuple[list[list[float]], int, int]:
    """Return (per-item label log-probabilities, number of fallback items, batch size used)."""
    pre, post = split_prompt(tok, d, demos)
    for it in items[:3]:                                   # the split must reproduce the real prompt exactly
        if pre + item_text(it) + post != render_template(tok, user_message(d, demos, it)):
            raise SystemExit("prefix/suffix split does not reproduce the rendered prompt")
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    fulls = tok([pre + item_text(it) + post for it in items], add_special_tokens=False)["input_ids"]
    P = len(pre_ids)
    suffixes, fallback = [], []
    for i, full in enumerate(fulls):
        if use_cache and len(full) > P and full[:P] == pre_ids:
            suffixes.append((i, full[P:]))
        else:
            fallback.append((i, full))
    B = batch_size
    while True:
        try:
            return _score(model, tok, pre_ids, suffixes, fallback, len(items), label_ids, B, device), len(fallback), B
        except RuntimeError as e:
            if not _is_oom(e) or B == 1:
                raise
            import torch
            torch.cuda.empty_cache()
            B = max(1, B // 2)
            log("llm", f"CUDA out of memory: retrying this job with batch size {B}")


def _score(model, tok, pre_ids, suffixes, fallback, n_items, label_ids, B, device):
    import torch
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    P = len(pre_ids)
    lab_index = [torch.tensor(label_ids[lab], device=device) for lab in LABELS]
    keep = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    scores: list = [None] * n_items

    def to_scores(last_logits):
        lp = torch.log_softmax(last_logits.float(), dim=-1)
        return torch.stack([torch.logsumexp(lp[:, ix], dim=-1) for ix in lab_index], dim=-1).cpu().tolist()

    def left_pad(seqs, L):
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, L - len(s):] = torch.tensor(s)
            mask[r, L - len(s):] = 1
        return ids.to(device), mask.to(device)

    with torch.inference_mode():
        if suffixes:
            order = sorted(suffixes, key=lambda x: -len(x[1]))            # longest first: OOM shows up at once
            Bc = min(B, len(order))
            base = model(input_ids=torch.tensor([pre_ids], device=device), use_cache=True).past_key_values
            croppable = getattr(base, "is_croppable", True) and hasattr(base, "crop") and not _FORCE_DEEPCOPY
            if croppable:
                cache = base
                cache.batch_repeat_interleave(Bc)
            for s in range(0, len(order), Bc):
                chunk = order[s:s + Bc]
                n_real = len(chunk)
                chunk = chunk + [chunk[0]] * (Bc - n_real)                 # constant batch size: cache stays valid
                L = max(len(x[1]) for x in chunk)
                ids, smask = left_pad([x[1] for x in chunk], L)
                mask = torch.cat([torch.ones((Bc, P), dtype=torch.long, device=device), smask], dim=1)
                pos = (P + smask.cumsum(-1) - 1).clamp(min=P)
                if not croppable:
                    cache = copy.deepcopy(base)
                    cache.batch_repeat_interleave(Bc)
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                               use_cache=True, **keep).logits
                out = to_scores(logits[:, -1])
                for r in range(n_real):
                    scores[chunk[r][0]] = out[r]
                if croppable:
                    cache.crop(P)
                else:
                    del cache
                del logits
            del base
        if fallback:
            fb = sorted(fallback, key=lambda x: -len(x[1]))
            Bf = max(1, B // 4)
            for s in range(0, len(fb), Bf):
                chunk = fb[s:s + Bf]
                L = max(len(x[1]) for x in chunk)
                ids, mask = left_pad([x[1] for x in chunk], L)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, **keep).logits
                out = to_scores(logits[:, -1])
                for r, (i, _) in enumerate(chunk):
                    scores[i] = out[r]
    if any(s is None for s in scores):
        raise RuntimeError("internal error: unscored items")
    return scores


# ----------------------------------------------------------------------------- main
def load_model(name: str, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    extra = {"device_map": str(device)} if device.type == "cuda" else {}  # load straight to the GPU (low CPU RAM)
    for kw in ({"dtype": dtype, **extra}, {"torch_dtype": dtype, **extra}, {"torch_dtype": dtype}):
        try:
            model = AutoModelForCausalLM.from_pretrained(name, **kw)
            break
        except (TypeError, ValueError, ImportError) as e:                   # older transformers / no accelerate
            log("llm", f"from_pretrained({sorted(kw)}) failed ({type(e).__name__}); trying the next form")
    else:
        raise SystemExit(f"could not load {name}")
    return tok, model.to(device).eval()


def signature(model_name: str, args, demos_sha: str) -> dict:
    return {"model": model_name, "scoring_version": SCORING_VERSION, "instruction_sha256": stable(INSTRUCTION),
            "max_items": args.max_items, "demonstrations_sha256": demos_sha, "k": K}


def write_job(odir: Path, j: dict, items: list[dict], scores, extra: dict) -> dict:
    import math
    rows, acc = [], defaultdict(lambda: [0, 0])
    for it, sc in zip(items, scores):
        pred = LABELS[max(range(3), key=lambda k: sc[k])]
        c = int(pred == it["label"])
        acc[it["set"]][0] += c
        acc[it["set"]][1] += 1
        mass = sum(math.exp(v) for v in sc)
        rows.append({"set": it["set"], "source_instance_id": it["source_instance_id"], "cluster_id": it["cluster_id"],
                     "gold": it["label"], "pred": pred, "correct": c,
                     **{f"s_{lab}": f"{v:.5f}" for lab, v in zip(LABELS, sc)}, "label_mass": f"{mass:.5f}"})
    odir.mkdir(parents=True, exist_ok=True)
    tmp = odir / f".{job_id(j)}.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(odir / f"{job_id(j)}.csv")
    summ = {**j, **extra, "accuracy": {k: v[0] / v[1] for k, v in acc.items()}, "n": {k: v[1] for k, v in acc.items()},
            "csv_sha256": sha256_file(odir / f"{job_id(j)}.csv")}
    atomic_write_json(odir / f"{job_id(j)}.DONE.json", summ)
    return summ


def preflight(model, tok, lab_ids, demos, evals, out_root: Path, name: str, batch_size: int, device,
              min_label_mass: float = 0.2) -> dict:
    """Real-model check before the long run: prompt, label tokens, cached vs uncached scores, speed."""
    import torch
    d = "restaurant"
    dem = demos[d]["Triplet__seed13"]
    by_set = defaultdict(list)
    for it in evals[d]:
        by_set[it["set"]].append(it)
    sample = [it for s in EVAL_SETS for it in by_set[s][:16]]
    t = time.time()
    cached, nfb, _ = score_job(model, tok, d, dem, sample, lab_ids, batch_size, True, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t
    full, _, _ = score_job(model, tok, d, dem, sample, lab_ids, max(1, batch_size // 4), False, device)
    a, b = torch.tensor(cached), torch.tensor(full)
    agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    pre, post = split_prompt(tok, d, dem)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    fulls = tok([pre + item_text(it) + post for it in evals[d]], add_special_tokens=False)["input_ids"]
    clean_split = sum(f[:len(pre_ids)] == pre_ids for f in fulls) / len(fulls)
    mass = torch.logsumexp(a, -1).exp()
    res = {"model": name, "label_tokens": {k: tok.convert_ids_to_tokens(v) for k, v in lab_ids.items()},
           "prefix_tokens": len(pre_ids), "share_items_with_clean_split": clean_split,
           "cached_vs_full_max_abs_diff": float((a - b).abs().max()), "cached_vs_full_argmax_agreement": agree,
           "label_mass_mean": float(mass.mean()), "items_per_second_cached": len(sample) / max(sec, 1e-9),
           "example_prompt": render_template(tok, user_message(d, dem, by_set["RevNon"][0])),
           "example_scores": dict(zip(LABELS, cached[2 * 16]))}
    atomic_write_json(out_root / "preflight" / f"{slug(name)}.json", res)
    log("preflight", f"{name}: prefix {len(pre_ids)} tokens; clean split {clean_split:.3f}; cached vs full: "
                     f"max |diff| {res['cached_vs_full_max_abs_diff']:.4f}, argmax agreement {agree:.3f}; "
                     f"label mass {res['label_mass_mean']:.3f}; ~{res['items_per_second_cached']:.0f} items/s")
    if agree < 0.9 or res["label_mass_mean"] < min_label_mass:
        raise SystemExit(f"preflight failed for {name}: {res}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="frozen Study-I project root (read only)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--shard", default="0/1", help="i/N: this notebook's share of the jobs")
    ap.add_argument("--subshard", default="0/1", help="j/M: this worker's share of the notebook's jobs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--max-items", type=int, default=None, help="per evaluation set; smoke tests only")
    ap.add_argument("--out-subdir", default=None, help=f"default {OUT_SUBDIR} ({OUT_SUBDIR}_smoke with --max-items)")
    ap.add_argument("--preflight", action="store_true", help="check each model on 64 items and exit (no job outputs)")
    ap.add_argument("--min-label-mass", type=float, default=0.2, help="preflight: minimum mean probability on labels")
    args = ap.parse_args(argv)
    import torch
    root = args.root.expanduser().resolve()
    out_root = root / (args.out_subdir or (OUT_SUBDIR + ("_smoke" if args.max_items else "")))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    demos = {d: build_demos(root, d) for d in DOMAINS}
    dpath = out_root / "demonstrations.json"
    if dpath.exists():
        if json.loads(dpath.read_text(encoding="utf-8")) != demos:
            raise SystemExit("demonstrations.json differs from the deterministic construction; refusing to mix runs")
    else:
        atomic_write_json(dpath, demos)
    demos_sha = stable(json.dumps(demos, sort_keys=True))
    evals = {d: load_eval(root, d, args.max_items) for d in DOMAINS}
    mine = select_jobs(jobs(models), args.shard, args.subshard)
    log("llm", f"shard {args.shard}, subshard {args.subshard}: {len(mine)} jobs; items per job "
               + ", ".join(f"{d} {len(evals[d])}" for d in DOMAINS) + f"; output {out_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32
    if args.preflight:
        for m in models:
            tok, model = load_model(m, dtype, device)
            preflight(model, tok, label_token_ids(tok), demos, evals, out_root, m, args.batch_size, device,
                      args.min_label_mass)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return 0
    by_model = defaultdict(list)
    for j in mine:
        by_model[j["model"]].append(j)
    t0 = time.time()
    for m, js in by_model.items():
        sig = signature(m, args, demos_sha)
        odir = out_root / slug(m)
        todo = []
        for j in js:
            done = odir / f"{job_id(j)}.DONE.json"
            if done.exists():
                old = json.loads(done.read_text(encoding="utf-8")).get("signature")
                if old != sig:
                    raise SystemExit(f"{done} was produced with a different configuration {old}; "
                                     f"refusing to overwrite (use another --out-subdir)")
                continue
            todo.append(j)
        if not todo:
            log("llm", f"{m}: all {len(js)} jobs of this worker are complete")
            continue
        log("llm", f"{m}: loading ({len(todo)} jobs to run, {len(js) - len(todo)} already done)")
        tok, model = load_model(m, dtype, device)
        lab_ids = label_token_ids(tok)
        log("llm", f"{m}: label tokens " + ", ".join(f"{k}={tok.convert_ids_to_tokens(v)}" for k, v in lab_ids.items()))
        for n, j in enumerate(todo, 1):
            d = j["domain"]
            dem = [] if j["condition"] == "ZS" else demos[d][f"{j['condition']}__seed{j['seed']}"]
            t1 = time.time()
            scores, n_fb, b_used = score_job(model, tok, d, dem, evals[d], lab_ids, args.batch_size,
                                             not args.no_prefix_cache, device)
            summ = write_job(odir, j, evals[d], scores, {
                "signature": sig, "fallback_items": n_fb, "batch_size": b_used, "seconds": round(time.time() - t1, 1),
                "prefix_cache": not args.no_prefix_cache, "dtype": str(dtype),
                "label_token_ids": lab_ids, "max_items": args.max_items})
            log("llm", f"{m} [{n}/{len(todo)}] {job_id(j)}: "
                       + ", ".join(f"{k} {v:.3f}" for k, v in summ["accuracy"].items())
                       + f" ({summ['seconds']} s, fallback {n_fb}; elapsed {(time.time() - t0) / 60:.1f} min)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log("llm", f"shard {args.shard}, subshard {args.subshard} finished in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
