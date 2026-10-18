#!/usr/bin/env python3
"""Hash-checked installation and rollback; run only with the server stopped."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

BUNDLE = Path(__file__).resolve().parent
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
BACKUP_NAME = ".prefix_retention_backup_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(prefix=".retention-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp defaults to owner-only. vLLM modules should remain readable.
        os.chmod(name, (path.stat().st_mode & 0o777) if path.exists() else 0o644)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_bundle():
    for name, expected in MANIFEST["payload_sha256"].items():
        path = BUNDLE / "payload" / name
        require(sha(path) == expected, f"Bundle payload changed: {path}")
        compile(path.read_bytes(), str(path), "exec")


def inspect(root):
    state = {}
    for rel, expected in MANIFEST["original_sha256"].items():
        actual = sha(root / rel)
        if actual == expected:
            state[rel] = "original"
        elif rel == "v1/core/block_pool.py" and actual == MANIFEST["payload_sha256"]["block_pool.py"]:
            state[rel] = "patched"
        else:
            state[rel] = "MISMATCH / missing"
    helper = root / "v1/core/prefix_retention.py"
    h = sha(helper)
    state["v1/core/prefix_retention.py"] = (
        "absent" if not helper.exists() else
        "patched" if h == MANIFEST["payload_sha256"]["prefix_retention.py"] else "MISMATCH")
    return state


def check_backup(backup):
    require(sha(backup / "block_pool.py") == MANIFEST["original_sha256"]["v1/core/block_pool.py"],
            "Backup missing or changed; will not restore unknown bytes")
    saved = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    require(saved == MANIFEST, "Backup belongs to a different patch bundle")


def install(root):
    state = inspect(root)
    for rel in MANIFEST["original_sha256"]:
        allowed = ("original", "patched") if rel == "v1/core/block_pool.py" else ("original",)
        require(state[rel] in allowed, f"Source differs from uploaded snapshot: {rel}")
    pool = root / "v1/core/block_pool.py"
    helper = root / "v1/core/prefix_retention.py"
    backup = root / BACKUP_NAME
    if state["v1/core/block_pool.py"] == "patched" and state["v1/core/prefix_retention.py"] == "patched":
        check_backup(backup)
        print("Already installed; no files changed.")
        return
    require(state["v1/core/block_pool.py"] == "original" and not helper.exists(),
            "Partial/unknown installation; use restore before reinstalling")
    if backup.exists():
        check_backup(backup)
    else:
        backup.mkdir()
        atomic_write(backup / "block_pool.py", pool.read_bytes())
        atomic_write(backup / "manifest.json", (BUNDLE / "manifest.json").read_bytes())
    try:
        atomic_write(helper, (BUNDLE / "payload/prefix_retention.py").read_bytes())
        atomic_write(pool, (BUNDLE / "payload/block_pool.py").read_bytes())
    except BaseException:
        # Best effort rollback; the verified backup also survives interruption.
        atomic_write(pool, (backup / "block_pool.py").read_bytes())
        if sha(helper) == MANIFEST["payload_sha256"]["prefix_retention.py"]:
            helper.unlink()
        raise
    print(f"Installed {MANIFEST['patch_id']} in {root}")
    print(f"Backup: {backup}; default policy remains lru. Restart the server.")


def restore(root):
    pool = root / "v1/core/block_pool.py"
    helper = root / "v1/core/prefix_retention.py"
    backup = root / BACKUP_NAME
    check_backup(backup)
    require(sha(pool) in (MANIFEST["original_sha256"]["v1/core/block_pool.py"],
                         MANIFEST["payload_sha256"]["block_pool.py"]),
            "block_pool.py changed since installation; refusing to overwrite")
    require(not helper.exists() or sha(helper) == MANIFEST["payload_sha256"]["prefix_retention.py"],
            "prefix_retention.py changed; refusing to remove it")
    atomic_write(pool, (backup / "block_pool.py").read_bytes())
    if helper.exists():
        helper.unlink()
    print("Original block_pool.py restored byte-for-byte; backup retained. Restart the server.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "install", "restore"])
    parser.add_argument("--vllm-root", type=Path, help="Optional explicit vllm package directory")
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        require(spec is not None and spec.submodule_search_locations,
                "vllm not found: use the server's Python environment/container")
        root = Path(next(iter(spec.submodule_search_locations)))
    root = root.resolve()
    validate_bundle()
    print(f"Target: {root}")
    if args.action == "check":
        state = inspect(root)
        print(json.dumps(state, indent=2))
        require(all("MISMATCH" not in s for s in state.values()), "Source check failed")
        pair = (state["v1/core/block_pool.py"], state["v1/core/prefix_retention.py"])
        require(pair in (("original", "absent"), ("patched", "patched")),
                "Incomplete installation; restore before continuing")
        if pair == ("patched", "patched"):
            check_backup(root / BACKUP_NAME)
    elif args.action == "install":
        install(root)
    else:
        restore(root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")
