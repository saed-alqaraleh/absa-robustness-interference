#!/usr/bin/env python3
"""Stage 1, one training run: any condition, domain, seed and hyperparameter configuration.

Fixed controls (protocol 3.7) follow the Study-I recipe: RoBERTa-base at the pinned
revision, AdamW (lr 2e-5, weight decay 0.01 except biases and LayerNorm), linear schedule
with 10% warmup over 8 epochs, batch 16, max length 128, fp16 autocast with FP32 master
weights and gradient scaling, gradient clipping 1.0, evaluation on the clean dev split
after every epoch, best clean-dev accuracy checkpoint, early stopping patience 2.

Only the added-record construction and the objective differ between methods. Grouped
methods place each source's clean anchor and its derivatives in the same batch; the
anchor is that clean record's single occurrence in the epoch, so every clean record is
still seen exactly once per epoch.

Phases: ``tuning`` (seed 101; evaluates the robustness-dev set, never ARTS),
``confirm`` (seeds 13/37/73; evaluates the ARTS sets) and ``regression`` (CE on the
Study-I Mixed file, to check this trainer against the Stage 0 reruns).

Restartable: a run whose DONE.json carries the same signature is skipped; a different
signature in an existing run directory is refused.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.common import (  # noqa: E402
    ARTS_FAMILIES, CONDITIONS, FEI_INFONCE, LABEL2ID, LABELS, PROTOCOL_SHA256, RECIPE, STAGE1_SUBDIR, TUNING_ONLY_CONTROLS,
    amendments, atomic_write_json, check_protocol, log, read_jsonl, sha256_file, sha256_text, write_csv,
)
from stage1 import losses as L  # noqa: E402

CODE_SHA = {f: sha256_file(Path(__file__).resolve().parent / f) for f in ("trainer.py", "losses.py", "common.py")}
DATA_DIR = {"clean_matched": "clean_matched", "mixed": "mixed", "bat": "bat", "fei": "fei", "triplet": "triplet"}


def spec_for(condition: str) -> dict:
    if condition in CONDITIONS:
        return CONDITIONS[condition]
    if condition in TUNING_ONLY_CONTROLS:
        return TUNING_ONLY_CONTROLS[condition]
    raise SystemExit(f"unknown condition {condition}")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_units(rows: list[dict], grouped: bool) -> list[list[int]]:
    """Training units: a source group (anchor first, then derivatives) or a single row."""
    if not grouped:
        return [[i] for i in range(len(rows))]
    derivs: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("is_augmented") and r.get("stage1_family") not in ("CleanRepeat",):
            derivs.setdefault(r["group"], []).append(i)
    units = []
    for i, r in enumerate(rows):
        if r.get("is_augmented"):
            if r.get("stage1_family") == "CleanRepeat":
                units.append([i])
            continue
        units.append([i] + derivs.pop(r["source_instance_id"], []))
    for orphan in derivs.values():          # derivative whose anchor is absent (should not happen)
        units.extend([[j] for j in orphan])
    return units


def make_batches(units: list[list[int]], batch_size: int, rng: random.Random) -> list[list[list[int]]]:
    """Shuffle units and pack them into batches of exactly `batch_size` items where possible.

    A unit that does not fit in the current batch is deferred to the next batch, and the
    current batch is topped up with later units that fit. Groups are never split, so
    grouped methods keep (almost) the same number of optimisation steps as plain batching.
    """
    order = list(range(len(units)))
    rng.shuffle(order)
    batches, cur, n, deferred = [], [], 0, []
    queue = order
    while queue or deferred:
        nxt = []
        for u in deferred + queue:
            size = len(units[u])
            if n + size <= batch_size:
                cur.append(units[u])
                n += size
                if n == batch_size:
                    batches.append(cur)
                    cur, n = [], 0
            else:
                nxt.append(u)
        if cur and nxt:                      # nothing else fits: close the partial batch
            batches.append(cur)
            cur, n = [], 0
        deferred, queue = nxt, []
    if cur:
        batches.append(cur)
    return batches


def encode(rows: list[dict], tok, max_length: int) -> list[dict]:
    enc = tok([r["sentence"] for r in rows], [r["aspect_term"] for r in rows], truncation=True, max_length=max_length)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(len(rows))]


def collate(batch_units, rows, feats, tok, device):
    import torch
    idx, group, anchor = [], [], []
    for g, unit in enumerate(batch_units):
        for k, i in enumerate(unit):
            idx.append(i)
            multi = len(unit) > 1
            group.append(g if multi else -1)
            anchor.append(1 if (multi and k == 0) else 0)
    padded = tok.pad([feats[i] for i in idx], return_tensors="pt")
    meta = {
        "labels": torch.tensor([LABEL2ID[rows[i]["label"]] for i in idx]),
        "orig_labels": torch.tensor([LABEL2ID[rows[i].get("orig_label", rows[i]["label"])] for i in idx]),
        "family": torch.tensor([L.FAMILY_CODE[rows[i].get("stage1_family", "Clean")] for i in idx]),
        "group": torch.tensor(group), "anchor": torch.tensor(anchor),
    }
    return {k: v.to(device) for k, v in padded.items()}, {k: v.to(device) for k, v in meta.items()}


def predict(model, rows, feats, tok, device, batch_size: int):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            b = tok.pad(feats[s:s + batch_size], return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**b).logits
            out.append(torch.softmax(logits.float(), -1).cpu())
    model.train()
    return torch.cat(out) if out else torch.zeros((0, 3))


def accuracy(rows, probs) -> float:
    if not rows:
        return float("nan")
    pred = probs.argmax(-1).tolist()
    return sum(LABEL2ID[r["label"]] == p for r, p in zip(rows, pred)) / len(rows)


def macro_f1(rows, probs) -> float:
    pred = probs.argmax(-1).tolist()
    gold = [LABEL2ID[r["label"]] for r in rows]
    f1s = []
    for c in range(3):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--domain", required=True, choices=("restaurant", "laptop"))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--config", default="{}", help="JSON hyperparameters for the method")
    ap.add_argument("--phase", required=True, choices=("tuning", "confirm", "regression"))
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--train-file", type=Path, default=None, help="regression phase only")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None, help="tests only")
    ap.add_argument("--model", default=RECIPE["model"], help="tests only")
    ap.add_argument("--revision", default=RECIPE["revision"], help="tests only")
    args = ap.parse_args()
    check_protocol()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    root = args.root.expanduser().resolve()
    spec = spec_for(args.condition)
    cfg = json.loads(args.config)
    data_root = root / STAGE1_SUBDIR / "data" / args.domain
    if args.phase == "regression":
        train_path = args.train_file or root / "transformations" / "materialized" / args.domain / "Clean+Mixed" / "train.jsonl"
    else:
        if not (root / STAGE1_SUBDIR / "data" / "_DATA_READY.json").exists():
            log("train", "ERROR: run build_data.py first (no _DATA_READY.json)")
            return 1
        train_path = data_root / DATA_DIR[spec["data"]] / "train.jsonl"
    dev_path = root / "data_processed" / args.domain / "dev.jsonl"
    if args.phase == "tuning":
        eval_paths = {"CleanDev": dev_path, **{f"Dev{f}": data_root / "robust_dev" / f"{f}.jsonl" for f in ("RT", "RN", "AD")}}
    else:
        eval_paths = {"Clean": root / "data_processed" / args.domain / "test.jsonl",
                      **{f: root / "data_processed" / args.domain / "evaluation" / f"{f}.jsonl" for f in ARTS_FAMILIES}}
    recipe = dict(RECIPE, model=args.model, revision=args.revision)
    if args.max_epochs:
        recipe["epochs"] = args.max_epochs
    signature = sha256_text(json.dumps({
        "condition": args.condition, "domain": args.domain, "seed": args.seed, "config": cfg, "phase": args.phase,
        "train_sha256": sha256_file(train_path), "dev_sha256": sha256_file(dev_path),
        "eval_sha256": {k: sha256_file(p) for k, p in eval_paths.items()}, "recipe": recipe,
        "protocol_sha256": PROTOCOL_SHA256, "spec": spec, "save_model": bool(args.save_model),
        "code_sha256": CODE_SHA, "amendments": amendments()}, sort_keys=True))
    out = args.out_dir.expanduser().resolve()
    done = out / "DONE.json"
    if done.exists():
        old = json.loads(done.read_text(encoding="utf-8"))
        if old.get("signature") == signature:
            log("train", f"SKIP (complete, same signature): {out.name}")
            return 0
        log("train", f"ERROR: {out} holds a different completed run; refusing to overwrite")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(recipe["model"], revision=recipe["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        recipe["model"], revision=recipe["revision"], num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID).float().to(device)
    rows = read_jsonl(train_path)
    if args.phase == "regression":   # Study-I files carry perturbation_family only
        for r in rows:
            r.setdefault("stage1_family", "Clean" if not r.get("is_augmented") else "CleanRepeat")
    dev_rows = read_jsonl(dev_path)
    feats = encode(rows, tok, recipe["max_length"])
    dev_feats = encode(dev_rows, tok, recipe["max_length"])
    method, grouped = spec["method"], spec["grouped"]
    units = make_units(rows, grouped)
    rng = random.Random(args.seed)
    epoch_batches = [make_batches(units, recipe["batch_size"], rng) for _ in range(recipe["epochs"])]
    steps_per_epoch = len(epoch_batches[0])
    total_steps = sum(len(b) for b in epoch_batches)      # exact schedule length
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm.weight", "layer_norm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm.weight", "layer_norm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": recipe["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=recipe["learning_rate"], eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(recipe["warmup_ratio"] * total_steps), total_steps)
    use_amp = recipe["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dro = L.GroupDROWeights(4, float(cfg.get("eta", 0.01))) if method == "groupdro" else None
    pc_gen = torch.Generator().manual_seed(args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    best_acc, best_state, best_epoch, bad, history = -1.0, None, -1, 0, []
    t0 = time.time()
    log("train", f"{out.name}: {len(rows)} rows, {len(units)} units, {steps_per_epoch} steps/epoch, method={method}, cfg={cfg}")
    model.train()
    for epoch in range(recipe["epochs"]):
        batches = epoch_batches[epoch]
        run_loss, aux_sum = 0.0, 0.0
        for step, bu in enumerate(batches):
            inputs, meta = collate(bu, rows, feats, tok, device)
            needs_rep = method in ("bat", "fei")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outp = model(**inputs, output_hidden_states=needs_rep)
            logits = outp.logits.float()
            ce = torch.nn.functional.cross_entropy(logits, meta["labels"])
            aux = torch.zeros((), device=device)
            if method == "scct":
                aux = cfg["lambda_i"] * L.invariance_loss(logits, meta["family"], meta["group"], meta["anchor"]) + \
                      cfg["lambda_s"] * L.sensitivity_loss(logits, meta["family"], meta["labels"], meta["orig_labels"], cfg.get("margin", 1.0))
            elif method == "consistency":
                aux = cfg["lambda_c"] * L.invariance_loss(logits, meta["family"], meta["group"], meta["anchor"])
            elif method == "bat":
                reps = outp.hidden_states[-1][:, 0, :]
                aux = cfg["lambda_bat"] * L.bat_triplet_loss(reps, meta["family"], meta["group"], meta["anchor"], cfg.get("margin", 1.0))
            elif method == "fei":
                reps = outp.hidden_states[-1][:, 0, :]
                aux = cfg["lambda_f"] * L.fei_opinion_loss(reps, meta["family"], meta["group"], meta["anchor"], cfg["mu"],
                                                           infonce=FEI_INFONCE)
            opt.zero_grad(set_to_none=True)
            if method == "groupdro":
                loss = dro.loss(L.per_family_ce(logits, meta["labels"], meta["family"]))
                scaler.scale(loss).backward()
            elif method == "pcgrad":
                gl = L.per_family_ce(logits, meta["labels"], meta["family"])
                n = sum(c for _, c in gl.values())
                task_losses = [lg * (c / n) for lg, c in gl.values()]
                loss = sum(task_losses)
                flat = []
                for k, tl in enumerate(task_losses):
                    gs = torch.autograd.grad(scaler.scale(tl), params, retain_graph=k < len(task_losses) - 1, allow_unused=True)
                    flat.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)]))
                combined = L.pcgrad_combine(flat, pc_gen) if len(flat) > 1 else flat[0]
                off = 0
                for p in params:
                    k = p.numel()
                    p.grad = combined[off:off + k].view_as(p).clone()
                    off += k
            else:
                loss = ce + aux
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, recipe["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss.detach())
            aux_sum += float(aux.detach())
        dev_acc = accuracy(dev_rows, predict(model, dev_rows, dev_feats, tok, device, recipe["eval_batch_size"]))
        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(1, len(batches)),
                        "aux_loss": aux_sum / max(1, len(batches)), "dev_accuracy": dev_acc})
        log("train", f"{out.name}: epoch {epoch + 1}/{recipe['epochs']} loss {history[-1]['train_loss']:.4f} "
                     f"aux {history[-1]['aux_loss']:.4f} dev_acc {dev_acc:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if dev_acc > best_acc:
            best_acc, best_epoch, bad = dev_acc, epoch + 1, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= recipe["early_stopping_patience"]:
                log("train", f"{out.name}: early stop after epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    eval_summary, pred_files = {}, {}
    for name, p in eval_paths.items():
        erows = read_jsonl(p)
        probs = predict(model, erows, encode(erows, tok, recipe["max_length"]), tok, device, recipe["eval_batch_size"])
        eval_summary[name] = {"accuracy": accuracy(erows, probs), "macro_f1": macro_f1(erows, probs), "n": len(erows)}
        pred = probs.argmax(-1).tolist()
        prow = [{"source_instance_id": r["source_instance_id"], "cluster_id": r.get("cluster_id", ""),
                 "gold": r["label"], "pred": LABELS[k], "correct": int(LABEL2ID[r["label"]] == k),
                 "p_negative": f"{pr[0]:.6f}", "p_neutral": f"{pr[1]:.6f}", "p_positive": f"{pr[2]:.6f}"}
                for r, k, pr in zip(erows, pred, probs.tolist())]
        write_csv(out / f"predictions_{name}.csv", prow)
        pred_files[name] = f"predictions_{name}.csv"
    if args.save_model:
        model.save_pretrained(out / "best_model")
        tok.save_pretrained(out / "best_model")
    import transformers
    result = {"condition": args.condition, "domain": args.domain, "seed": args.seed, "config": cfg, "phase": args.phase,
              "method": method, "grouped": grouped, "train_file": str(train_path), "train_rows": len(rows),
              "best_epoch": best_epoch, "best_dev_accuracy": best_acc, "history": history,
              "eval_summary": eval_summary, "prediction_files": pred_files, "runtime_min": (time.time() - t0) / 60,
              "protocol_sha256": PROTOCOL_SHA256, "recipe": recipe, "code_sha256": CODE_SHA, "amendments": amendments(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": transformers.__version__,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    atomic_write_json(out / "run_result.json", result)
    atomic_write_json(done, {"status": "complete", "signature": signature,
                             "result_sha256": sha256_file(out / "run_result.json")})
    log("train", f"{out.name}: done in {(time.time() - t0) / 60:.1f} min; "
                 + ", ".join(f"{k} {v['accuracy']:.4f}" for k, v in eval_summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
