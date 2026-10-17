from __future__ import annotations
import json, os, platform, subprocess, sys, time
from pathlib import Path
from typing import Any

from .hashing import sha256_file, hash_object


def file_hash_or_none(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return sha256_file(p) if p.exists() and p.is_file() else None


def git_code_version(root: str | Path) -> str:
    root = Path(root)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(root), "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) != 0
        return commit + ("+dirty" if dirty else "")
    except Exception:
        # Stable fallback over source/scripts/config files, avoiding caches and outputs.
        files = []
        for rel in ("src", "scripts", "configs", "pyproject.toml", "requirements.txt"):
            p = root / rel
            if p.is_file():
                files.append((str(p.relative_to(root)), sha256_file(p)))
            elif p.exists():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".pyc"):
                        files.append((str(f.relative_to(root)), sha256_file(f)))
        return "tree-sha256:" + hash_object(files)


def collect_environment() -> dict[str, Any]:
    obj: dict[str, Any] = {
        "time_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch
        obj.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as e:
        obj["torch_error"] = repr(e)
    try:
        import transformers
        obj["transformers"] = transformers.__version__
    except Exception as e:
        obj["transformers_error"] = repr(e)
    try:
        import sklearn
        obj["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return obj


def append_jsonl_locked(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object with an advisory lock on POSIX runtimes (e.g. Colab)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
