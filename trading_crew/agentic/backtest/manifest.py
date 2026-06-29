"""Run manifest: capture every reproducibility-relevant input (paper §8.3).

A reproducible backtest result needs more than just the equity curve.
The paper §8.3 lists the minimum manifest:

1. Code identity         — git SHA (and a flag if the tree is dirty)
2. Prompt identity       — SHA-256 of the agent prompts in use
3. RNG seed              — for the simulator and any randomised sizing
4. Cost parameters       — fee / spread / impact constants
5. Data hashes           — SHA-256 of every OHLCV cache file that fed
                           the backtest, so a future replay can detect
                           silent data revisions
6. LLM identity          — provider, model, temperature
7. Timestamp / host

The manifest is written next to the backtest results in JSON and is
loaded by the UI's "replay" link.  Two runs with identical manifests
*must* produce identical equity curves; if they don't, you have a
non-determinism leak somewhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


@dataclass
class RunManifest:
    """Everything you need to replay a backtest run byte-for-byte.

    Fields are deliberately flat so a manifest can be diffed across runs
    with a simple ``diff -u`` after loading the JSON.
    """

    run_id: str
    created_ts: str
    code_git_sha: str
    code_dirty: bool
    prompts_hash: str
    seed: int
    cost_params: Dict[str, float]
    data_hashes: Dict[str, str]
    llm_provider: str
    llm_model: str
    llm_temperature: float
    host: str
    platform: str
    extra: Dict[str, str] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        version = data.get("manifest_version", 0)
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"RunManifest version mismatch: file has {version}, "
                f"code expects {MANIFEST_VERSION}"
            )
        # Pass through every field that exists; future versions can add
        # fields without breaking older readers as long as the version is bumped.
        kwargs = {k: data[k] for k in (
            "run_id", "created_ts", "code_git_sha", "code_dirty",
            "prompts_hash", "seed", "cost_params", "data_hashes",
            "llm_provider", "llm_model", "llm_temperature",
            "host", "platform",
        )}
        kwargs["extra"] = data.get("extra", {})
        kwargs["manifest_version"] = version
        return cls(**kwargs)


def _git_sha(repo_root: Path) -> tuple[str, bool]:
    """Return ``(sha, is_dirty)`` for the repo containing ``repo_root``.

    Returns ``("unknown", False)`` if not in a git checkout — the manifest
    is still useful as a snapshot of the rest of the environment.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ("unknown", False)
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = False
    return (sha, dirty)


def _hash_file(path: Path, chunk: int = 1 << 16) -> str:
    """SHA-256 of a file's bytes; returns ``""`` if the file is missing."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_data_dir(directory: Path, glob: str = "*.csv") -> Dict[str, str]:
    """Hash every file in ``directory`` matching ``glob`` (default OHLCV CSVs).

    Returns a mapping of filename -> sha256.  Used to capture every
    OHLCV cache file that fed the backtest so a future replay can verify
    nothing on disk has been silently re-fetched with a different vendor
    cut.
    """
    if not directory.is_dir():
        return {}
    return {p.name: _hash_file(p) for p in sorted(directory.glob(glob))}


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(
    *,
    run_id: str,
    repo_root: Path,
    seed: int,
    prompts_hash: str,
    cost_params: Dict[str, float],
    data_hashes: Dict[str, str],
    llm_provider: str,
    llm_model: str,
    llm_temperature: float,
    extra: Optional[Dict[str, str]] = None,
) -> RunManifest:
    """Construct a manifest, querying git + host info automatically."""
    sha, dirty = _git_sha(repo_root)
    return RunManifest(
        run_id=run_id,
        created_ts=datetime.now(timezone.utc).isoformat(),
        code_git_sha=sha,
        code_dirty=dirty,
        prompts_hash=prompts_hash,
        seed=seed,
        cost_params=dict(cost_params),
        data_hashes=dict(data_hashes),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        host=socket.gethostname(),
        platform=f"{platform.system()} {platform.release()} python-{platform.python_version()}",
        extra=dict(extra or {}),
    )


def write_manifest(manifest: RunManifest, path: str | os.PathLike) -> None:
    """Atomically write ``manifest`` to ``path`` as JSON.

    Uses the same atomic-write pattern as the portfolio state store: a
    ``.tmp`` next to the destination, then ``os.replace``.  A partial
    write never leaves a broken manifest on disk.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(manifest.to_dict(), fh, indent=2, sort_keys=True)
        os.replace(tmp, out)
    except OSError as exc:
        logger.warning("could not persist manifest to %s: %s", out, exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def load_manifest(path: str | os.PathLike) -> RunManifest:
    """Read a manifest JSON back into a ``RunManifest``."""
    with open(path, "r", encoding="utf-8") as fh:
        return RunManifest.from_dict(json.load(fh))
