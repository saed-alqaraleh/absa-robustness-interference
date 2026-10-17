#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from absa_xfer.campaign import (
    build_plan,
    get_stage_design,
    make_train_command,
    write_plan_csv,
)


def _load_pins(path: Path | None) -> dict:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    pins = obj.get("pins", obj)
    if not isinstance(pins, dict):
        raise ValueError(f"Invalid model-pins structure in {path}")
    return pins


def build_campaign(
    config_path: str | Path,
    stage: str,
    out_dir: str | Path,
    model_pins_path: str | Path | None = None,
) -> dict:
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    design = get_stage_design(cfg, stage)
    pins = _load_pins(Path(model_pins_path) if model_pins_path else None)

    # Final scientific stages must never float Hugging Face revisions.
    if stage in {"minimal", "full", "lopo"}:
        missing_pins = [
            model
            for model in design["models"]
            if not (pins.get(model) or {}).get("revision")
        ]
        if missing_pins:
            raise RuntimeError(
                f"Immutable model revision(s) missing for stage={stage}: {missing_pins}. "
                "Provide --model-pins from the frozen model-pin manifest."
            )

    rows = build_plan(
        cfg["data"]["domains"],
        design["models"],
        design["conditions"],
        design["seeds"],
        pins,
    )

    expected = (
        len(cfg["data"]["domains"])
        * len(design["models"])
        * len(design["conditions"])
        * len(design["seeds"])
    )
    if len(rows) != expected or len({r["run_id"] for r in rows}) != expected:
        raise AssertionError(
            f"Campaign cardinality/uniqueness failure: rows={len(rows)}, expected={expected}"
        )
    if stage == "lopo" and expected != 18:
        raise AssertionError(f"Frozen minimal LOPO stage must contain 18 runs, found {expected}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = out / f"{stage}_plan.csv"
    write_plan_csv(rows, plan)

    hp = {
        **cfg["training"],
        "max_length": cfg["models"]["max_length"],
        "pair_order": cfg["models"].get("pair_order", "sentence_first"),
    }
    commands = [
        make_train_command(row, cfg["project"]["root"], hp, stage)
        for row in rows
    ]
    if len(commands) != len(set(commands)):
        raise AssertionError("Generated campaign contains duplicate command strings")

    sh = out / f"{stage}_commands.sh"
    sh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    sh.chmod(0o755)

    return {
        "stage": stage,
        "n_runs": len(rows),
        "plan": str(plan),
        "commands": str(sh),
        "models": design["models"],
        "conditions": design["conditions"],
        "seeds": design["seeds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stage", choices=["minimal", "full", "lopo"], default="minimal")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--model-pins",
        default=None,
        help="JSON from 00_pin_models.py. Required for scientific campaign stages.",
    )
    args = ap.parse_args()

    result = build_campaign(args.config, args.stage, args.out_dir, args.model_pins)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
