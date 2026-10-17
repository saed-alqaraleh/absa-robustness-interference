#!/usr/bin/env python3
"""One Study IV run: LoRA fine-tuning of Qwen3-8B on one frozen Study I training file, or the zero-shot
reference (condition ZS, no training), followed by scoring of the clean test split and the protected
ARTS cohorts.

Restartable:
  * a run whose DONE.json carries the same signature is skipped; a DONE.json with a different
    signature is never overwritten (exit code 2);
  * after every epoch the adapter, optimizer, scheduler and RNG states are saved to <out>/_resume/,
    so an interrupted run continues from the last completed epoch. The resume folder is deleted
    once DONE.json is written.

Outputs in <out>/: predictions.csv (one row per evaluation item), run_result.json, best_adapter.pt
(LoRA weights of the selected epoch, unless --no-keep-adapter), DONE.json, train.log is written by the
caller (run_plan.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora.common import (  # noqa: E402
    ARTS_FAMILIES, KIT_VERSION, LABELS, LORA, MODEL_ID, OUT_SUBDIR, RECIPE, ZS, atomic_write_json, code_digests,
    dev_path, eval_paths, log, protocol_sha256, run_name, sha256_file, stable, train_path,
)
from lora.data import label_counts, load_dev, load_eval, load_train  # noqa: E402
from lora.scoring import encode, label_index, label_scores, label_token_ids, left_pad, pad_id, score_items  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_lambda_factory(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def load_model(name: str, revision: str | None, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"           # `dtype` replaced `torch_dtype` in 4.56
    extra = {"device_map": str(device)} if device.type == "cuda" else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, **{dkey: dtype}, **extra)
    except (ValueError, ImportError) as e:                                    # e.g. accelerate missing for device_map
        log("model", f"from_pretrained with device_map failed ({type(e).__name__}: {e}); loading without it")
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, **{dkey: dtype})
    model = model.to(device)
    got = next(model.parameters()).dtype
    if got != dtype:
        raise SystemExit(f"model loaded in {got}, expected {dtype}; check the transformers version")
    return tok, model


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("torch", "transformers", "peft", "accelerate"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def j_si(items: list[dict], correct: list[int]) -> tuple[float | None, int]:
    by = {f: {} for f in ("RevTgt", "RevNon")}
    for it, c in zip(items, correct):
        if it["set"] in by:
            by[it["set"]][it["source_instance_id"]] = c
    common = sorted(set(by["RevTgt"]) & set(by["RevNon"]))
    if not common:
        return None, 0
    return sum(by["RevTgt"][s] * by["RevNon"][s] for s in common) / len(common), len(common)


def write_predictions(out: Path, items: list[dict], scores: list[list[float]]) -> tuple[list[int], dict, dict]:
    rows, correct = [], []
    acc, n = {}, {}
    for it, sc in zip(items, scores):
        pred = LABELS[max(range(3), key=lambda k: sc[k])]
        c = int(pred == it["label"])
        correct.append(c)
        acc[it["set"]] = acc.get(it["set"], 0) + c
        n[it["set"]] = n.get(it["set"], 0) + 1
        rows.append({"set": it["set"], "source_instance_id": it["source_instance_id"], "cluster_id": it["cluster_id"],
                     "gold": it["label"], "pred": pred, "correct": c,
                     **{f"s_{lab}": f"{v:.6f}" for lab, v in zip(LABELS, sc)},
                     "label_mass": f"{sum(math.exp(v) for v in sc):.6f}"})
    tmp = out / ".predictions.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out / "predictions.csv")
    return correct, {k: acc[k] / n[k] for k in acc}, n


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="Study I project root (read only)")
    ap.add_argument("--domain", required=True, choices=("restaurant", "laptop"))
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=None, help="pinned Hugging Face commit (required for real runs)")
    ap.add_argument("--allow-unpinned", action="store_true", help="tests only")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--no-keep-adapter", action="store_true")
    # Overrides for smoke tests only; every one is part of the signature.
    ap.add_argument("--max-train-items", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--expected-train-rows", type=int, default=None, help="tests only (0 disables the check)")
    ap.add_argument("--no-eval-count-check", action="store_true", help="tests only")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    root = args.root.expanduser().resolve()
    j = {"domain": args.domain, "condition": args.condition, "seed": args.seed}
    out = (args.out_dir or root / OUT_SUBDIR / "runs" / run_name(j)).expanduser().resolve()
    if args.revision is None and not args.allow_unpinned:
        raise SystemExit("--revision is required (pinned model commit); see run_plan.py preflight")
    recipe = dict(RECIPE)
    if args.max_epochs is not None:
        recipe["max_epochs"] = args.max_epochs
    is_zs = args.condition == ZS

    sig = {"kit_version": KIT_VERSION, "protocol_sha256": protocol_sha256(), "code": code_digests(),
           "model": args.model, "revision": args.revision, **j, "recipe": recipe, "lora": None if is_zs else LORA,
           "train_sha256": None if is_zs else sha256_file(train_path(root, args.domain, args.condition)),
           "dev_sha256": sha256_file(dev_path(root, args.domain)),
           "eval_sha256": {k: sha256_file(p) for k, p in eval_paths(root, args.domain).items()},
           "overrides": {"max_train_items": args.max_train_items, "max_epochs": args.max_epochs,
                         "expected_train_rows": args.expected_train_rows, "eval_count_check": not args.no_eval_count_check}}
    signature = stable(json.dumps(sig, sort_keys=True))
    done = out / "DONE.json"
    if done.exists():
        old = json.loads(done.read_text(encoding="utf-8")).get("signature")
        if old == signature:
            log("run", f"SKIP (complete, same signature): {out.name}")
            return 0
        log("run", f"ERROR: {out} holds a completed run with a different signature; refusing to overwrite")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "signature.json", {"signature": signature, **sig})

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    set_seed(args.seed)

    evals = load_eval(root, args.domain, check_counts=not args.no_eval_count_check)
    dev = load_dev(root, args.domain)
    train = [] if is_zs else load_train(root, args.domain, args.condition, args.expected_train_rows)
    if args.max_train_items:
        train = train[:args.max_train_items]

    tok, model = load_model(args.model, args.revision, device, dtype)
    lab_ids = label_token_ids(tok)
    resolved_rev = getattr(model.config, "_commit_hash", None)
    if args.revision and resolved_rev and resolved_rev != args.revision:
        raise SystemExit(f"loaded commit {resolved_rev} differs from the pinned revision {args.revision}")
    max_tok = recipe["max_prompt_tokens"]
    eval_enc = encode(tok, args.domain, evals, max_tok)
    dev_enc = encode(tok, args.domain, dev, max_tok)
    log("run", f"{out.name}: {len(train)} training rows {label_counts(train) if train else ''}; "
               f"dev {len(dev)}; eval {len(evals)}; device {device}; dtype {dtype}")

    history, best_epoch, best_dev = [], None, None
    train_seconds = 0.0
    if not is_zs:
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        if recipe["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.config.use_cache = False
        torch.manual_seed(args.seed)                                    # LoRA initialisation
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        group = recipe["micro_batch"] * recipe["grad_accum"]
        steps_per_epoch = math.ceil(len(train) / group)
        total = steps_per_epoch * recipe["max_epochs"]
        warmup = math.ceil(recipe["warmup_ratio"] * total)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total, warmup))
        train_enc = encode(tok, args.domain, train, max_tok)
        y_all = torch.tensor([LABELS.index(r["label"]) for r in train], dtype=torch.long)
        lab_index = label_index(lab_ids, device)
        pad = pad_id(tok)

        resume = out / "_resume" / "state.pt"
        best_path = out / "best_adapter.pt"
        start_epoch, bad = 1, 0
        if resume.exists():
            st = torch.load(resume, map_location="cpu", weights_only=False)
            if st["signature"] != signature:
                raise SystemExit(f"{resume} belongs to a different configuration; delete it deliberately to restart")
            set_peft_model_state_dict(model, st["adapter"])
            opt.load_state_dict(st["optimizer"])
            sched.load_state_dict(st["scheduler"])
            torch.set_rng_state(st["rng_cpu"])
            if device.type == "cuda" and st.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(st["rng_cuda"])
            history, best_epoch, best_dev, bad = st["history"], st["best_epoch"], st["best_dev"], st["bad"]
            train_seconds = st["train_seconds"]
            start_epoch = st["epoch_done"] + 1
            log("run", f"RESUME from epoch {st['epoch_done']} (best dev {best_dev:.4f} at epoch {best_epoch})")
        log("run", f"LoRA trainable parameters {n_train_params:,}; {steps_per_epoch} optimizer steps per epoch; "
                   f"warm-up {warmup} of {total} planned steps")

        stop = bad >= recipe["patience"]
        for epoch in range(start_epoch, recipe["max_epochs"] + 1):
            if stop:
                break
            model.train()
            t0 = time.time()
            g = torch.Generator().manual_seed(args.seed * 100003 + epoch)
            perm = torch.randperm(len(train), generator=g).tolist()
            loss_sum, seen = 0.0, 0
            for step in range(steps_per_epoch):
                idx = perm[step * group:(step + 1) * group]
                for m in range(0, len(idx), recipe["micro_batch"]):
                    mb = idx[m:m + recipe["micro_batch"]]
                    ids, mask = left_pad([train_enc[i] for i in mb], pad, device)
                    scores = label_scores(model, ids, mask, lab_index)
                    loss = F.cross_entropy(scores, y_all[mb].to(device), reduction="sum") / len(idx)
                    loss.backward()
                    loss_sum += loss.item() * len(idx)
                    seen += len(mb)
                torch.nn.utils.clip_grad_norm_(params, recipe["max_grad_norm"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if (step + 1) % 25 == 0 or step + 1 == steps_per_epoch:
                    el = time.time() - t0
                    eta = el / (step + 1) * (steps_per_epoch - step - 1)
                    log("train", f"{out.name} epoch {epoch}/{recipe['max_epochs']} step {step + 1}/{steps_per_epoch} "
                                 f"loss {loss_sum / max(1, seen):.4f} lr {sched.get_last_lr()[0]:.2e} "
                                 f"elapsed {el / 60:.1f} min, epoch ETA {eta / 60:.1f} min")
            dev_scores, _ = score_items(model, tok, dev_enc, lab_ids, args.eval_batch, device)
            dev_acc = sum(int(LABELS[max(range(3), key=lambda k: s[k])] == it["label"])
                          for s, it in zip(dev_scores, dev)) / len(dev)
            secs = time.time() - t0
            train_seconds += secs
            history.append({"epoch": epoch, "dev_accuracy": dev_acc, "train_loss": loss_sum / max(1, seen),
                            "seconds": round(secs, 1)})
            if best_dev is None or dev_acc > best_dev:
                best_dev, best_epoch, bad = dev_acc, epoch, 0
                atomic_torch_save({k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(model).items()},
                                  best_path)
            else:
                bad += 1
            log("train", f"{out.name} epoch {epoch}: dev accuracy {dev_acc:.4f} (best {best_dev:.4f} at epoch {best_epoch})")
            stop = bad >= recipe["patience"]
            atomic_torch_save({"signature": signature, "epoch_done": epoch, "history": history,
                               "best_epoch": best_epoch, "best_dev": best_dev, "bad": bad,
                               "train_seconds": train_seconds,
                               "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                               "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                               "rng_cpu": torch.get_rng_state(),
                               "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                              resume)
        # evaluate the selected epoch, loaded from the stored adapter (what is released is what is scored)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, best)
        log("run", f"{out.name}: training done; evaluating epoch {best_epoch} (dev {best_dev:.4f})")
    else:
        n_train_params = 0

    t_eval = time.time()
    scores, used_batch = score_items(model, tok, eval_enc, lab_ids, args.eval_batch, device)
    correct, acc, n = write_predictions(out, evals, scores)
    jsi, n_j = j_si(evals, correct)
    fam = [acc[f] for f in ARTS_FAMILIES]
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    result = {**j, "model": args.model, "revision": args.revision, "resolved_commit": resolved_rev,
              "accuracy": acc, "n": n, "R_mean": sum(fam) / 3, "W": min(fam), "J_SI": jsi, "n_J": n_j,
              "best_epoch": best_epoch, "best_dev_accuracy": best_dev, "history": history,
              "train_rows": len(train), "train_label_counts": label_counts(train) if train else {},
              "trainable_parameters": n_train_params, "train_seconds": round(train_seconds, 1),
              "eval_seconds": round(time.time() - t_eval, 1), "eval_batch_used": used_batch,
              "total_seconds": round(time.time() - t_start, 1), "gpu": gpu, "versions": versions(),
              "signature": signature}
    atomic_write_json(out / "run_result.json", result)
    if args.no_keep_adapter and (out / "best_adapter.pt").exists():
        (out / "best_adapter.pt").unlink()
    atomic_write_json(done, {"status": "complete", "signature": signature,
                             "predictions_sha256": sha256_file(out / "predictions.csv"),
                             "result_sha256": sha256_file(out / "run_result.json")})
    shutil.rmtree(out / "_resume", ignore_errors=True)
    log("run", f"{out.name}: DONE in {(time.time() - t_start) / 60:.1f} min; "
               + ", ".join(f"{k} {v:.4f}" for k, v in acc.items()) + (f", J_SI {jsi:.4f}" if jsi is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
