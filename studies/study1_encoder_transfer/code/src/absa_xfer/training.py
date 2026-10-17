from __future__ import annotations
import json, os, random, time, inspect
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import LABEL2ID, classification_metrics
from .hashing import hash_object
from .io_utils import atomic_write_json, ensure_dir

def trainer_processing_kwargs(trainer_cls, tokenizer):
    """Return the processor keyword supported by this Transformers Trainer API."""
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    raise RuntimeError(
        "Unsupported transformers.Trainer API: neither processing_class nor tokenizer is accepted"
    )

def adapt_training_arguments_kwargs(training_args_cls, kwargs: dict) -> dict:
    """Preserve frozen TrainingArguments semantics across Transformers v4/v5.

    Known API translations only:
      * eval_strategy <-> evaluation_strategy
      * warmup_ratio -> warmup_steps for v5, where a float <1 is interpreted
        by Transformers as a ratio of total training steps.

    Any remaining unsupported parameter fails closed.
    """
    params = inspect.signature(training_args_cls.__init__).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    out = dict(kwargs)

    if "eval_strategy" in out and "eval_strategy" not in params:
        if "evaluation_strategy" in params:
            out["evaluation_strategy"] = out.pop("eval_strategy")
        else:
            raise RuntimeError(
                "Unsupported TrainingArguments API: neither eval_strategy nor "
                "evaluation_strategy is accepted"
            )

    if "warmup_ratio" in out and "warmup_ratio" not in params:
        if "warmup_steps" in params:
            ratio = float(out.pop("warmup_ratio"))
            if not (0.0 <= ratio < 1.0):
                raise RuntimeError(
                    f"warmup_ratio must be in [0,1) for v5 mapping; got {ratio}"
                )
            out["warmup_steps"] = ratio
        else:
            raise RuntimeError(
                "Unsupported TrainingArguments API: warmup_ratio is unavailable "
                "and warmup_steps is not accepted"
            )

    if not accepts_var_kwargs:
        unsupported = sorted(k for k in out if k not in params)
        if unsupported:
            raise RuntimeError(
                "Unsupported TrainingArguments parameters after compatibility "
                f"mapping: {unsupported}"
            )
    return out

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

class PairDataset:
    def __init__(self, rows, tokenizer, max_length=128, pair_order="sentence_first"):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pair_order = pair_order
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        first, second = (r["sentence"], r["aspect_term"]) if self.pair_order=="sentence_first" else (r["aspect_term"],r["sentence"])
        enc = self.tokenizer(first, second, truncation=True, max_length=self.max_length)
        enc["labels"] = LABEL2ID[r["label"]]
        return enc

def train_one_run(
    train_rows: list[dict],
    dev_rows: list[dict],
    test_sets: dict[str,list[dict]],
    model_name: str,
    output_dir: str | Path,
    seed: int,
    hp: dict,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    pair_order: str = "sentence_first",
    run_signature: str | None = None,
):
    """
    Hugging Face training entry point. This function intentionally imports heavy
    dependencies lazily so provenance/audit tests can run on CPU-only machines.
    """
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
    )
    import torch

    set_global_seed(seed)
    out = ensure_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=tokenizer_revision or model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=model_revision, num_labels=3,
        id2label={0:"negative",1:"neutral",2:"positive"},
        label2id={"negative":0,"neutral":1,"positive":2}
    )
    # ABSA_AMP_MASTER_WEIGHTS_FP32_V1
    # Hugging Face Trainer fp16 uses AMP/GradScaler. Trainable master
    # weights must therefore remain FP32. Some released checkpoints,
    # notably microsoft/deberta-v3-base, contain FP16 tensors.
    # Casting here preserves fp16 AMP while standardizing master weights
    # across BERT, RoBERTa, and DeBERTa.
    model = model.float()
    _trainable_dtypes = {
        p.dtype for p in model.parameters() if p.requires_grad
    }
    if _trainable_dtypes != {torch.float32}:
        raise RuntimeError(
            "AMP master-weight compatibility failure: expected all "
            f"trainable parameters to be FP32, got {_trainable_dtypes}"
        )

    tr_ds = PairDataset(train_rows,tokenizer,hp["max_length"],pair_order)
    dv_ds = PairDataset(dev_rows,tokenizer,hp["max_length"],pair_order)
    collator = DataCollatorWithPadding(tokenizer)

    def cm(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits,axis=-1)
        return classification_metrics(labels,preds)

    args_kwargs = dict(
        output_dir=os.environ.get("ABSA_HF_CHECKPOINT_DIR", str(out/"hf")),
        learning_rate=float(hp["learning_rate"]),
        per_device_train_batch_size=int(hp["train_batch_size"]),
        per_device_eval_batch_size=int(hp.get("eval_batch_size",hp["train_batch_size"])),
        num_train_epochs=float(hp["epochs"]),
        weight_decay=float(hp.get("weight_decay",0.01)),
        warmup_ratio=float(hp.get("warmup_ratio",0.1)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        seed=seed,
        data_seed=seed,
        fp16=bool(hp.get("fp16",False) and torch.cuda.is_available()),
        bf16=bool(hp.get("bf16",False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        report_to=[],
        logging_steps=max(1,int(hp.get("logging_steps",20))),
    )
    args_kwargs = adapt_training_arguments_kwargs(TrainingArguments, args_kwargs)
    targs = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,args=targs,train_dataset=tr_ds,eval_dataset=dv_ds,
        data_collator=collator,compute_metrics=cm,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(hp.get("early_stopping_patience",2)))],
        **trainer_processing_kwargs(Trainer, tokenizer),
    )
    start=time.time()
    result=trainer.train()
    elapsed=time.time()-start
    trainer.save_model(str(out/"best_model"))
    tokenizer.save_pretrained(str(out/"best_model"))

    prediction_files={}
    eval_summary={}
    for name, rows in test_sets.items():
        ds=PairDataset(rows,tokenizer,hp["max_length"],pair_order)
        pred=trainer.predict(ds)
        y_pred=np.argmax(pred.predictions,axis=-1)
        y_true=pred.label_ids
        metrics=classification_metrics(y_true,y_pred)
        eval_summary[name]=metrics
        p_rows=[]
        for r,yt,yp in zip(rows,y_true,y_pred):
            p_rows.append({
                "example_id":r.get("example_id",r["source_instance_id"]+"|"+name),
                "source_instance_id":r["source_instance_id"],
                "cluster_id":r["cluster_id"],
                "y_true":int(yt),"y_pred":int(yp),
                "label_true":r["label"],"evaluation_condition":name
            })
        pth=out/f"predictions_{name}.csv"
        pd.DataFrame(p_rows).to_csv(pth,index=False)
        prediction_files[name]=str(pth)

    state={
        "seed":seed,"model_name":model_name,"model_revision":model_revision,
        "pair_order":pair_order,"hyperparameters":hp,
        "train_runtime_seconds":elapsed,
        "trainer_metrics":result.metrics,
        "eval_summary":eval_summary,
        "prediction_files":prediction_files,
        "run_signature":run_signature,
    }
    atomic_write_json(out/"run_result.json",state)
    atomic_write_json(out/"DONE.json",{"status":"complete","result_hash":hash_object(state),"run_signature":run_signature})
    return state
