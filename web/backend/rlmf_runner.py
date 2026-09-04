"""Backend controller for the RLMF training + serving controls in the UI.

This module is the HTTP layer's single entry point for everything the
"RLMF" panel in the RL-Training tab needs:

* **Episodes / dataset** — read the resolved-episode training data
  (``build_dataset``), the raw episode store, and a summary the UI
  tabulates.  This is the "expose the training data" surface.
* **LoRA fine-tune** — kick off :func:`rlmf_lora.run_finetune` on a
  background worker thread against *any* base model the user picks in
  the dropdown, streaming step logs + honouring a cooperative Stop.
* **Serving** — launch / stop ``scripts/vllm_serve.sh`` with or without
  the trained LoRA adapter, and register the served endpoint(s) as
  runtime LLM presets so the operator can select "…(with LoRA)" or
  "…(base)" in the run-config dropdown for *inference*.

Design mirrors :mod:`web.backend.rl_runner` (single background job,
in-memory status registry, on-disk artefacts) so the two training
surfaces feel the same to the UI.  Heavy ML imports stay lazy inside
the worker so importing this module (and thus starting the web app)
never pulls in torch / transformers.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import traceback
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Sanitise a payload for ``JSONResponse`` (strict ``json.dumps``).

    FastAPI's ``JSONResponse`` rejects non-finite floats (``ValueError: Out
    of range float values are not JSON compliant: nan``) — which a NaN
    ``realised_return`` / reward (unresolved or missing price) used to 500
    the whole RLMF endpoint on every page load.  Non-finite floats become
    ``None`` so the UI renders "—" instead of the tab failing to load.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    return Path(
        os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    )


def _default_output_dir() -> Path:
    """Where the LoRA adapter lands — same convention as
    ``scripts/vllm_serve.sh`` so "train then serve" is one env."""
    return Path(
        os.environ.get("LORA_OUTPUT_DIR")
        or (_cache_dir() / "rlmf" / "lora_adapter")
    )


def _adapter_dir(output_dir: Optional[Path] = None) -> Path:
    return (output_dir or _default_output_dir()) / "adapter"


def _repo_root() -> Path:
    # web/backend/rlmf_runner.py -> agentic_workflow/
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Episodes / dataset (read-only surfaces)
# ---------------------------------------------------------------------------


def _episode_store_path() -> Path:
    return _cache_dir() / "memory" / "episodes.jsonl"


def list_episodes(
    *,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Return raw episodes (newest first) — the source data RLMF trains on.

    ``status`` filters by outcome status (PENDING / RESOLVED / ABANDONED).
    ``state_summary`` is truncated for the table; the full text is still
    available through the dataset endpoint / on disk.
    """
    from trading_crew.agentic.memory import EpisodicMemory

    store = _episode_store_path()
    mem = EpisodicMemory(store)
    eps = mem.all_episodes()

    sym = (symbol or "").strip().upper()
    st = (status or "").strip().upper()
    rows: List[Dict[str, Any]] = []
    counts = {"PENDING": 0, "RESOLVED": 0, "ABANDONED": 0}
    sample_count = 0
    for ep in eps:
        counts[ep.outcome_status.value] = counts.get(ep.outcome_status.value, 0) + 1
        if ep.episode_id.startswith(SAMPLE_PREFIX):
            sample_count += 1
        if sym and ep.symbol.upper() != sym:
            continue
        if st and ep.outcome_status.value != st:
            continue
        proposal = ep.action_proposal or {}
        rows.append(
            {
                "episode_id": ep.episode_id,
                "symbol": ep.symbol,
                "decision_ts": ep.decision_ts,
                "outcome_ts": ep.outcome_ts,
                "outcome_status": ep.outcome_status.value,
                "regime": ep.regime.value,
                "action": str(proposal.get("side") or proposal.get("action") or "").upper(),
                "conviction_score": proposal.get("conviction_score"),
                "horizon_days": proposal.get("horizon_days"),
                "realised_return": ep.realised_return,
                "alpha_return": ep.alpha_return,
                "max_drawdown": ep.max_drawdown,
                "reflection": ep.reflection,
                "state_summary": (ep.state_summary or "")[:600],
                "retrieval_count": getattr(ep, "retrieval_count", 0),
            }
        )
    # Newest decisions first.
    rows.sort(key=lambda r: r["decision_ts"], reverse=True)
    return _json_safe({
        "total": len(eps),
        "counts": counts,
        "sample_count": sample_count,
        "returned": min(len(rows), limit),
        "store_path": str(mem.db_path),
        "store_exists": mem.db_path.exists(),
        "episodes": rows[:limit],
    })


# Sample episodes are tagged with this episode_id prefix so they can be
# seeded for a demo and cleanly removed again without touching real data.
SAMPLE_PREFIX = "SAMPLE-"


def seed_sample_episodes(*, count: int = 12, symbol: str = "SAMPLE.NS") -> Dict[str, Any]:
    """Write a spread of synthetic *resolved* episodes so the RLMF feature
    can be exercised end-to-end before any real decision has resolved.

    The rows deliberately span tiers (SHORT/MEDIUM/LONG), sides
    (BUY/SELL/HOLD), convictions and win/loss outcomes so the reward
    ranking, dataset summary, and few-shot pool all look meaningful.
    Every row is tagged ``SAMPLE-…`` and can be removed via
    :func:`clear_sample_episodes` — real episodes are never touched.
    """
    from datetime import timedelta

    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime

    count = max(1, min(int(count), 200))
    sym = (symbol or "SAMPLE.NS").strip().upper()

    # (horizon_days, side, conviction, realised, alpha, regime)
    templates = [
        (1260, "BUY", 0.85, 0.62, 0.34, Regime.TREND),      # LONG hi-conviction win
        (1260, "SELL", 0.80, -0.41, -0.29, Regime.CRISIS),  # LONG hi-conviction loss
        (180, "BUY", 0.62, 0.17, 0.09, Regime.TREND),       # MEDIUM solid win
        (180, "HOLD", 0.45, 0.01, -0.02, Regime.RANGE),     # MEDIUM flat
        (90, "BUY", 0.70, 0.11, 0.04, Regime.TREND),        # MEDIUM win
        (90, "SELL", 0.55, -0.08, -0.05, Regime.HIGH_VOL),  # MEDIUM loss
        (21, "BUY", 0.35, 0.04, 0.01, Regime.RANGE),        # SHORT weak win
        (21, "SELL", 0.40, -0.06, -0.03, Regime.HIGH_VOL),  # SHORT loss
    ]

    store = _episode_store_path()
    mem = EpisodicMemory(store)
    base_dt = datetime(2025, 1, 6, tzinfo=timezone.utc)
    added = 0
    for i in range(count):
        hd, side, conv, realised, alpha, regime = templates[i % len(templates)]
        decision_dt = base_dt + timedelta(days=i * 7)
        mem.add(
            Episode(
                episode_id=f"{SAMPLE_PREFIX}{sym}-{i:03d}",
                symbol=sym,
                decision_ts=decision_dt.isoformat(),
                state_summary=(
                    f"[SAMPLE DATA] Synthetic {side} thesis on {sym} — "
                    f"tier={hd}d, conviction={conv:.2f}. Generated for RLMF "
                    "feature testing; not a real market observation."
                ),
                regime=regime,
                action_proposal={
                    "side": side,
                    "conviction_score": conv,
                    "horizon_days": hd,
                    "target_weight": round(0.01 + 0.02 * conv, 4),
                },
                outcome_ts=(decision_dt + timedelta(days=hd)).isoformat(),
                outcome_status=OutcomeStatus.RESOLVED,
                realised_return=realised,
                alpha_return=alpha,
                max_drawdown=-abs(realised) * 0.4,
                reflection=(
                    "Sample lesson: thesis played out as modelled."
                    if realised >= 0
                    else "Sample lesson: conviction outran the evidence."
                ),
            )
        )
        added += 1
    return {"seeded": added, "symbol": sym, "store_path": str(mem.db_path)}


def clear_sample_episodes() -> Dict[str, Any]:
    """Remove every ``SAMPLE-…`` episode, leaving real data untouched."""
    from trading_crew.agentic.memory import EpisodicMemory

    store = _episode_store_path()
    mem = EpisodicMemory(store)
    if not mem.db_path.exists():
        return {"removed": 0}
    sample_ids = [e.episode_id for e in mem.all_episodes()
                  if e.episode_id.startswith(SAMPLE_PREFIX)]
    for eid in sample_ids:
        mem.remove(eid)
    return {"removed": len(sample_ids)}


def dataset_summary(*, symbol: Optional[str] = None) -> Dict[str, Any]:
    """RLMF dataset rows (resolved episodes) + aggregate summary."""
    from trading_crew.agentic.rlmf import build_dataset, summarise_dataset

    sym = (symbol or "").strip() or None
    rows = build_dataset(symbol=sym)
    return _json_safe({
        "symbol": sym,
        "summary": summarise_dataset(rows),
        "rows": [r.to_jsonl_dict() for r in rows],
    })


def export_dataset(*, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Write the FinGPT-Forecaster-compatible JSONL to disk."""
    from trading_crew.agentic.rlmf import build_dataset, default_dataset_path, export_dataset as _export

    sym = (symbol or "").strip() or None
    rows = build_dataset(symbol=sym)
    path = _export(rows, default_dataset_path())
    return {"path": str(path), "rows": len(rows), "symbol": sym}


# ---------------------------------------------------------------------------
# LoRA fine-tune — background job
# ---------------------------------------------------------------------------


@dataclass
class _LoraJob:
    job_id: str
    base_model: str
    output_dir: str
    dataset_path: str
    status: str = "running"  # running | completed | stopped | failed
    started_ts: float = field(default_factory=time.time)
    finished_ts: Optional[float] = None
    adapter_dir: Optional[str] = None
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "dataset_path": self.dataset_path,
            "status": self.status,
            "active": self.status == "running",
            "started_ts": self.started_ts,
            "elapsed_sec": (self.finished_ts or time.time()) - self.started_ts,
            "adapter_dir": self.adapter_dir,
            "error": self.error,
            "config": self.config,
            "logs": self.logs[-200:],
        }


_LORA_LOCK = threading.RLock()
_LORA_JOB: Optional[_LoraJob] = None


def start_lora_training(
    *,
    base_model: str,
    symbol: Optional[str] = None,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_len: int = 2048,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Export the dataset then fine-tune a LoRA adapter on ``base_model``.

    Single-tenant: refuses to start when a fine-tune is already running
    (one GPU, one job).  Returns the job snapshot immediately; poll
    :func:`get_lora_status`.
    """
    global _LORA_JOB
    base_model = (base_model or "").strip()
    if not base_model:
        raise ValueError("base_model is required")

    with _LORA_LOCK:
        if _LORA_JOB is not None and _LORA_JOB.status == "running":
            raise ValueError(
                f"A fine-tune is already running (job {_LORA_JOB.job_id}). "
                "Stop it before starting another."
            )

    # Export the dataset first so training always sees the freshest
    # resolved episodes.  Fails loudly (no rows) rather than training
    # on a stale file.
    export = export_dataset(symbol=symbol)
    if export["rows"] == 0:
        raise ValueError(
            "No resolved episodes to train on yet — the RLMF dataset is "
            "empty.  Run analyses and let their outcome windows resolve "
            "(Memory tab → Resolve outcomes) before fine-tuning."
        )

    out_dir = Path(output_dir) if output_dir else _default_output_dir()
    job_id = f"lora-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    job = _LoraJob(
        job_id=job_id,
        base_model=base_model,
        output_dir=str(out_dir),
        dataset_path=export["path"],
        config={
            "base_model": base_model,
            "symbol": symbol,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "max_seq_len": max_seq_len,
            "dataset_rows": export["rows"],
        },
    )

    def _worker() -> None:
        from trading_crew.agentic.rlmf_lora import run_finetune

        try:
            adapter = run_finetune(
                dataset_path=Path(job.dataset_path),
                output_dir=out_dir,
                base_model=base_model,
                epochs=epochs,
                learning_rate=learning_rate,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_seq_len=max_seq_len,
                should_stop=job.stop_flag.is_set,
                on_log=lambda payload: job.logs.append(
                    {**payload, "ts": datetime.now(timezone.utc).isoformat()}
                ),
            )
            if job.stop_flag.is_set():
                job.status = "stopped"
            else:
                job.status = "completed"
            job.adapter_dir = str(adapter)
        except SystemExit as exc:  # rlmf_lora raises SystemExit on bad input
            job.status = "failed"
            job.error = str(exc)
        except Exception as exc:
            logger.exception("RLMF LoRA fine-tune failed (%s)", job_id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
        finally:
            job.finished_ts = time.time()

    thread = threading.Thread(target=_worker, name=f"rlmf-{job_id}", daemon=True)
    job.thread = thread
    with _LORA_LOCK:
        _LORA_JOB = job
    thread.start()
    return job.snapshot()


def stop_lora_training() -> Dict[str, Any]:
    with _LORA_LOCK:
        job = _LORA_JOB
    if job is None or job.status != "running":
        return {"stopped": False, "reason": "no active fine-tune"}
    job.stop_flag.set()
    return {"stopped": True, "job_id": job.job_id}


def get_lora_status() -> Dict[str, Any]:
    with _LORA_LOCK:
        job = _LORA_JOB
    if job is None:
        return {"active": False, "status": "idle", "job_id": None, "logs": []}
    return job.snapshot()


def list_adapters() -> Dict[str, Any]:
    """Discover trained adapters on disk (the default output dir)."""
    adapter = _adapter_dir()
    ready = (adapter / "adapter_model.safetensors").exists()
    base = None
    cfg = adapter / "adapter_config.json"
    if cfg.exists():
        try:
            import json

            base = json.loads(cfg.read_text()).get("base_model_name_or_path")
        except Exception:
            base = None
    return {
        "default_output_dir": str(_default_output_dir()),
        "adapter_dir": str(adapter),
        "ready": ready,
        "base_model": base,
    }


# ---------------------------------------------------------------------------
# vLLM serving — subprocess management
# ---------------------------------------------------------------------------


@dataclass
class _ServerProc:
    mode: str  # "with_lora" | "base"
    base_model: str
    lora_name: str
    port: int
    base_url: str
    log_path: str
    proc: subprocess.Popen
    started_ts: float = field(default_factory=time.time)

    def snapshot(self) -> Dict[str, Any]:
        alive = self.proc.poll() is None
        return {
            "running": alive,
            "mode": self.mode,
            "base_model": self.base_model,
            "lora_name": self.lora_name if self.mode == "with_lora" else None,
            "port": self.port,
            "base_url": self.base_url,
            "pid": self.proc.pid,
            "exit_code": self.proc.returncode,
            "log_path": self.log_path,
            "elapsed_sec": time.time() - self.started_ts,
        }


_SERVER_LOCK = threading.RLock()
_SERVER: Optional[_ServerProc] = None

_RUNTIME_PRESET_PREFIX = "served-"


def _register_serving_presets(srv: _ServerProc) -> None:
    """Expose the served endpoint(s) in the run-config LLM dropdown."""
    from trading_crew.llm_presets import (
        LlmPreset,
        clear_runtime_presets,
        register_runtime_preset,
    )

    clear_runtime_presets(prefix=_RUNTIME_PRESET_PREFIX)
    register_runtime_preset(
        LlmPreset(
            id=f"{_RUNTIME_PRESET_PREFIX}base",
            label=f"Served · {srv.base_model} (base)",
            kind="open-source",
            provider="hosted_vllm",
            model=srv.base_model,
            base_url=srv.base_url,
            api_env="OPENAI_API_KEY",
            description="Base model served locally via scripts/vllm_serve.sh (no LoRA).",
            tags=["served", "vLLM", "base"],
        )
    )
    if srv.mode == "with_lora":
        register_runtime_preset(
            LlmPreset(
                id=f"{_RUNTIME_PRESET_PREFIX}lora",
                label=f"Served · {srv.base_model} + RLMF LoRA",
                kind="open-source",
                provider="hosted_vllm",
                model=srv.lora_name,
                base_url=srv.base_url,
                api_env="OPENAI_API_KEY",
                description="Base model + RLMF LoRA adapter served locally (inference with LoRA).",
                tags=["served", "vLLM", "lora"],
            )
        )


def start_server(
    *,
    mode: str = "with_lora",
    base_model: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    port: int = 8091,
    lora_name: str = "rlmf",
    max_model_len: Optional[int] = None,
    gpu_mem_util: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch ``scripts/vllm_serve.sh`` in a child process.

    ``mode`` is ``"with_lora"`` (serve base + adapter, selectable at
    inference) or ``"base"`` (base weights only).  Single managed
    server — starting a new one stops the previous.  Registers runtime
    LLM presets so the served model shows up in the run dropdown.
    """
    global _SERVER
    mode = (mode or "with_lora").lower()
    if mode not in ("with_lora", "base"):
        raise ValueError("mode must be 'with_lora' or 'base'")

    base_model = (base_model or os.environ.get("BASE_MODEL") or "Qwen/Qwen3-32B").strip()

    with _SERVER_LOCK:
        if _SERVER is not None and _SERVER.proc.poll() is None:
            _stop_server_locked()

    script = _repo_root() / "scripts" / "vllm_serve.sh"
    if not script.exists():
        raise ValueError(f"serve script not found: {script}")

    cmd: List[str] = ["bash", str(script)]
    cmd.append("--with-lora" if mode == "with_lora" else "--without-lora")
    cmd += ["--base-model", base_model, "--port", str(port), "--lora-name", lora_name]
    if adapter_dir:
        cmd += ["--adapter", str(Path(adapter_dir))]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if gpu_mem_util:
        cmd += ["--gpu-mem-util", str(gpu_mem_util)]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if tool_parser:
        cmd += ["--tool-parser", tool_parser]

    log_dir = _cache_dir() / "rlmf" / "serve_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
    log_f = log_path.open("w", encoding="utf-8")
    log_f.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
    log_f.flush()

    # New session so we can signal the whole process group on stop
    # (vLLM spawns children).
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(_repo_root()),
        start_new_session=True,
    )
    base_url = f"http://localhost:{port}/v1"
    srv = _ServerProc(
        mode=mode,
        base_model=base_model,
        lora_name=lora_name,
        port=port,
        base_url=base_url,
        log_path=str(log_path),
        proc=proc,
    )
    with _SERVER_LOCK:
        _SERVER = srv
    _register_serving_presets(srv)
    return srv.snapshot()


def _stop_server_locked() -> None:
    global _SERVER
    srv = _SERVER
    if srv is None:
        return
    if srv.proc.poll() is None:
        try:
            os.killpg(os.getpgid(srv.proc.pid), signal.SIGTERM)
            try:
                srv.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(srv.proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception("failed to stop vLLM server")
    from trading_crew.llm_presets import clear_runtime_presets

    clear_runtime_presets(prefix=_RUNTIME_PRESET_PREFIX)
    _SERVER = None


def stop_server() -> Dict[str, Any]:
    with _SERVER_LOCK:
        if _SERVER is None:
            return {"stopped": False, "reason": "no server running"}
        _stop_server_locked()
    return {"stopped": True}


def _probe_models(base_url: str) -> Optional[List[str]]:
    """Best-effort check that the OpenAI-compatible endpoint is up."""
    try:
        import httpx

        r = httpx.get(f"{base_url}/models", timeout=2.0)
        if r.status_code == 200:
            return [m.get("id") for m in (r.json().get("data") or [])]
    except Exception:
        return None
    return None


def get_server_status() -> Dict[str, Any]:
    with _SERVER_LOCK:
        srv = _SERVER
    if srv is None:
        return {"running": False, "ready": False}
    snap = srv.snapshot()
    if not snap["running"]:
        # Process died — surface the tail of the log so the UI can show why.
        snap["ready"] = False
        try:
            snap["log_tail"] = Path(srv.log_path).read_text(errors="replace")[-3000:]
        except Exception:
            snap["log_tail"] = None
        return snap
    served = _probe_models(srv.base_url)
    snap["ready"] = served is not None
    snap["served_models"] = served
    return snap
