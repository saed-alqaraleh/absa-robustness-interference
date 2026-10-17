from __future__ import annotations
import csv, json, re
from pathlib import Path
from typing import Any
from .hashing import sha256_file

FAMILY_PATTERNS = {
    "RevTgt": re.compile(r"rev[_-]?tgt|reverse[_-]?target", re.I),
    "RevNon": re.compile(r"rev[_-]?non|reverse[_-]?non", re.I),
    "AddDiff": re.compile(r"add[_-]?diff|different[_-]?aspect", re.I),
    "Clean": re.compile(r"(^|[/_.-])(orig(inal)?|clean|test)([/_.-]|$)", re.I),
}
SUPPORTED = {".xml", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".raw"}


def infer_family_from_path(path: str | Path) -> str | None:
    s = str(path).replace("\\", "/")
    # A transformed file may naturally be named "test.*" inside a RevTgt/RevNon/AddDiff
    # directory. Perturbation evidence therefore has priority over generic Clean/test hints.
    perturb_hits = [fam for fam in ("RevTgt", "RevNon", "AddDiff") if FAMILY_PATTERNS[fam].search(s)]
    if len(perturb_hits) == 1:
        return perturb_hits[0]
    if len(perturb_hits) > 1:
        return None
    return "Clean" if FAMILY_PATTERNS["Clean"].search(s) else None


def _peek_text(path: Path, n: int = 3) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [next(f).rstrip("\n") for _ in range(n)]
    except (StopIteration, UnicodeDecodeError, OSError):
        try:
            return path.read_text(encoding="utf-8", errors="replace").splitlines()[:n]
        except Exception:
            return []


def discover_arts_files(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    if not root.exists():
        raise FileNotFoundError(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        rows.append({
            "relative_path": str(p.relative_to(root)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "family_hint": infer_family_from_path(p.relative_to(root)),
            "suffix": p.suffix.lower(),
            "head": _peek_text(p, 3),
        })
    grouped = {fam: [r["relative_path"] for r in rows if r["family_hint"] == fam]
               for fam in ("Clean", "RevTgt", "RevNon", "AddDiff")}
    unresolved = [r["relative_path"] for r in rows if r["family_hint"] is None]
    return {
        "repo_root": str(root),
        "n_candidate_files": len(rows),
        "files": rows,
        "family_candidates": grouped,
        "unresolved_files": unresolved,
        "automatic_family_mapping_safe": all(len(grouped[f]) == 1 for f in ("RevTgt", "RevNon", "AddDiff")),
        "note": (
            "Discovery is intentionally non-destructive. Do not infer scientific lineage from filenames alone; "
            "an explicit mapping must be frozen before evaluation ingestion."
        ),
    }
