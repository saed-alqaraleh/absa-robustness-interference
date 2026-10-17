#!/usr/bin/env python3
"""Study IV driver: preflight, run queue (resumable, shardable across Colab sessions) and status.

  python lora/run_plan.py preflight --root ROOT          # inputs, model pin, GPU checks, timing
  python lora/run_plan.py run       --root ROOT --shard 0/1
  python lora/run_plan.py status    --root ROOT

Each run is a separate process (lora/train_lora.py), so GPU memory is released between runs. Output is
streamed live and copied to <root>/lora_qwen3_v1/logs/<run>.log. Re-running `run` skips completed runs
and resumes an interrupted run from its last completed epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lora.common import (  # noqa: E402
    CONDITIONS, DOMAINS, EXPECTED_EVAL, KIT_DIR, KIT_VERSION, LABELS, LORA, MODEL_ID, RECIPE, ZS, atomic_write_json, dev_path,
    eval_paths, jobs, log, out_root, protocol_sha256, run_name, select_jobs, sha256_file, train_path,
)
from lora.data import label_counts, load_dev, load_eval, load_train  # noqa: E402

PY = sys.executable


def revision_file(root: Path) -> Path:
    return out_root(root) / "MODEL_REVISION.json"


def pinned_revision(root: Path) -> str | None:
    p = revision_file(root)
    return json.loads(p.read_text())["revision"] if p.exists() else None


# ----------------------------------------------------------------------------- preflight
def preflight(args) -> int:
    root = args.root
    report = {"kit_version": KIT_VERSION, "protocol_sha256": protocol_sha256(), "model": args.model, "inputs": {}}
    for d in DOMAINS:
        ev = load_eval(root, d)
        dev = load_dev(root, d)
        tr = {c: load_train(root, d, c) for c in CONDITIONS}
        report["inputs"][d] = {
            "eval_counts": {s: sum(1 for x in ev if x["set"] == s) for s in ("Clean", "RevTgt", "RevNon", "AddDiff")},
            "dev": len(dev), "train_rows": {c: len(v) for c, v in tr.items()},
            "train_labels": {c: label_counts(v) for c, v in tr.items()},
            "sha256": {**{f"train/{c}": sha256_file(train_path(root, d, c)) for c in CONDITIONS},
                       "dev": sha256_file(dev_path(root, d)),
                       **{f"eval/{k}": sha256_file(p) for k, p in eval_paths(root, d).items()}}}
        log("preflight", f"{d}: inputs OK; eval {report['inputs'][d]['eval_counts']}; dev {len(dev)}; "
                         f"train rows {report['inputs'][d]['train_rows']}")

    local = Path(args.model).is_dir()                    # a local model folder (tests): no hub look-up
    rev = pinned_revision(root)
    if rev is None:
        rev = args.revision
        if rev is None and local:
            raise SystemExit("a local model folder needs an explicit --revision label")
        if rev is None:
            from huggingface_hub import HfApi
            rev = HfApi().model_info(args.model).sha
        atomic_write_json(revision_file(root), {"model": args.model, "revision": rev,
                                                "pinned_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        log("preflight", f"pinned {args.model} at commit {rev}")
    elif args.revision and args.revision != rev:
        raise SystemExit(f"MODEL_REVISION.json pins {rev}; refusing to switch to {args.revision}")
    report["revision"] = rev

    if not local:
        from huggingface_hub import snapshot_download
        t0 = time.time()
        snapshot_download(args.model, revision=rev)
        log("preflight", f"model files ready in the local cache ({(time.time() - t0) / 60:.1f} min)")

    if not args.skip_gpu_checks:
        report["gpu_checks"] = gpu_checks(root, args.model, rev)
    atomic_write_json(out_root(root) / "preflight" / "PREFLIGHT.json", report)
    log("preflight", f"PASS. Report: {out_root(root) / 'preflight' / 'PREFLIGHT.json'}")
    return 0


def gpu_checks(root: Path, model_name: str, rev: str, device=None) -> dict:
    import torch
    from lora.scoring import encode, label_index, label_scores, label_token_ids, left_pad, pad_id, render, score_items
    from lora.common import user_message
    from lora.train_lora import load_model
    if device is None:
        if not torch.cuda.is_available():
            raise SystemExit("no GPU visible: select an A100 runtime (or pass --skip-gpu-checks to test the inputs only)")
        device = torch.device("cuda")
    d = DOMAINS[0]
    tok, model = load_model(model_name, rev, device, torch.bfloat16)
    lab = label_token_ids(tok)
    dev = load_dev(root, d)[:64]
    enc = encode(tok, d, dev, RECIPE["max_prompt_tokens"])
    sc, _ = score_items(model, tok, enc, lab, 32, device)
    import math
    mass = [sum(math.exp(v) for v in s) for s in sc]
    zs_acc = sum(int(LABELS[max(range(3), key=lambda k: s[k])] == it["label"]) for s, it in zip(sc, dev)) / len(dev)
    out = {"gpu": torch.cuda.get_device_name(0), "label_token_ids": lab,
           "label_tokens_decoded": {k: [tok.decode([i]) for i in v] for k, v in lab.items()},
           "example_prompt": render(tok, user_message(d, dev[0])),
           "zero_shot_dev64_accuracy": zs_acc, "mean_label_mass": sum(mass) / len(mass), "min_label_mass": min(mass),
           "max_prompt_tokens_seen": max(len(x) for x in enc)}
    log("preflight", f"label tokens {out['label_tokens_decoded']}; zero-shot dev(64) accuracy {zs_acc:.3f}; mean label mass {out['mean_label_mass']:.3f}")
    # timing: six optimizer steps of the real recipe on 96 training items (nothing is saved)
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    train = load_train(root, d, "Clean+Mixed")[:96]
    tenc = encode(tok, d, train, RECIPE["max_prompt_tokens"])
    y = torch.tensor([LABELS.index(r["label"]) for r in train], device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RECIPE["learning_rate"])
    li = label_index(lab, device)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mb = RECIPE["micro_batch"]
    for s in range(0, 96, mb * RECIPE["grad_accum"]):
        for m in range(s, s + mb * RECIPE["grad_accum"], mb):
            ids, mask = left_pad(tenc[m:m + mb], pad_id(tok), device)
            loss = torch.nn.functional.cross_entropy(label_scores(model, ids, mask, li), y[m:m + mb])
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_step = (time.time() - t0) / 6
    peak = torch.cuda.max_memory_allocated() / 2**30
    ev_t0 = time.time()
    score_items(model, tok, enc, lab, 32, device)
    per_item_eval = (time.time() - ev_t0) / len(enc)
    est = {}
    for dd, rows in (("restaurant", 4455), ("laptop", 2581)):
        steps = math.ceil(rows / (mb * RECIPE["grad_accum"]))
        n_eval = sum(EXPECTED_EVAL[dd].values())
        n_dev = len(load_dev(root, dd))
        est[dd] = round((steps * per_step + n_dev * per_item_eval) * RECIPE["max_epochs"] / 60
                        + n_eval * per_item_eval / 60 + 2.0, 1)          # + about 2 min model loading
    total_h = round(sum(est[dd] for dd in est) * len(CONDITIONS) * 3 / 60, 1)
    out.update({"seconds_per_optimizer_step": round(per_step, 3), "peak_memory_gib": round(peak, 1),
                "minutes_per_run_upper_bound": est, "total_hours_upper_bound_one_gpu": total_h})
    log("preflight", f"{per_step:.2f} s per optimizer step; peak memory {peak:.1f} GiB; "
                     f"upper bound per run {est} min; all 24 runs about {total_h} h on one GPU (less with early stopping)")
    del model, opt
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- run queue
def _stream(proc, log_fh, name: str, heartbeat: int) -> None:
    last = [time.time()]

    def beat():
        while proc.poll() is None:
            time.sleep(5)
            if time.time() - last[0] > heartbeat:
                print(f"[{time.strftime('%H:%M:%S')}] [heartbeat] {name} still running", flush=True)
                last[0] = time.time()
    threading.Thread(target=beat, daemon=True).start()
    for line in proc.stdout:
        last[0] = time.time()
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        log_fh.flush()


def run(args) -> int:
    root = args.root
    rev = pinned_revision(root)
    if rev is None or not (out_root(root) / "preflight" / "PREFLIGHT.json").exists():
        raise SystemExit("run the preflight first (it pins the model revision)")
    mine = select_jobs(jobs(), args.shard)
    logs = out_root(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = []
    log("run", f"shard {args.shard}: {len(mine)} jobs; model {args.model} @ {rev[:12]}")
    for k, j in enumerate(mine, 1):
        name = run_name(j)
        if (out_root(root) / "runs" / name / "DONE.json").exists() and not args.recheck:
            log("run", f"[{k}/{len(mine)}] {name}: complete (DONE.json present)")
            continue
        log("run", f"[{k}/{len(mine)}] {name}: starting (elapsed {(time.time() - t0) / 60:.1f} min)")
        cmd = [PY, "-u", str(KIT_DIR / "lora" / "train_lora.py"), "--root", str(root), "--domain", j["domain"],
               "--condition", j["condition"], "--seed", str(j["seed"]), "--model", args.model, "--revision", rev,
               "--eval-batch", str(args.eval_batch)] + (["--no-keep-adapter"] if args.no_keep_adapter else []) \
            + (args.train_args.split() if args.train_args else [])
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with (logs / f"{name}.log").open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            _stream(proc, fh, name, args.heartbeat)
            rc = proc.wait()
        if rc != 0:
            failed.append((name, rc))
            log("run", f"[{k}/{len(mine)}] {name}: FAILED (exit {rc}); see {logs / (name + '.log')}")
            if not args.keep_going:
                return rc
        else:
            log("run", f"[{k}/{len(mine)}] {name}: finished")
    log("run", f"shard {args.shard}: {len(mine) - len(failed)} of {len(mine)} jobs ok in {(time.time() - t0) / 60:.1f} min")
    return 1 if failed else 0


def status(args) -> int:
    rows = []
    for j in jobs():
        name = run_name(j)
        out = out_root(args.root) / "runs" / name
        if (out / "DONE.json").exists():
            r = json.loads((out / "run_result.json").read_text())
            acc = r["accuracy"]
            state = (f"done  Clean {acc['Clean']:.3f}  RT {acc['RevTgt']:.3f}  RN {acc['RevNon']:.3f}  "
                     f"AD {acc['AddDiff']:.3f}  J_SI {r['J_SI']:.3f}  epoch {r['best_epoch']}")
        elif (out / "_resume" / "state.pt").exists():
            state = "in progress (resumable)"
        elif out.exists():
            state = "started"
        else:
            state = "pending"
        rows.append((name, state))
    n_done = sum(1 for _, s in rows if s.startswith("done"))
    print(f"{n_done} of {len(rows)} jobs complete")
    for name, s in rows:
        print(f"  {name:40s} {s}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("preflight", "run", "status"))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=None, help="pin this commit at preflight (default: current main)")
    ap.add_argument("--shard", default="0/1", help="i/N: this notebook's share of the jobs")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--recheck", action="store_true", help="let train_lora.py verify signatures of completed runs")
    ap.add_argument("--no-keep-adapter", action="store_true")
    ap.add_argument("--heartbeat", type=int, default=120)
    ap.add_argument("--skip-gpu-checks", action="store_true")
    ap.add_argument("--train-args", default="", help="tests only: extra train_lora.py options (part of each signature)")
    args = ap.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    return {"preflight": preflight, "run": run, "status": status}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
