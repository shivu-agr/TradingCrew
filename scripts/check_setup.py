#!/usr/bin/env python3
"""Offline setup checks for a fresh clone (no LLM / no live market calls).

Usage (from the project root, inside the venv):

    python scripts/check_setup.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str) -> None:
    print(f"  ok   {msg}")


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  warn {msg}")


def check_python() -> None:
    v = sys.version_info
    if v < (3, 10):
        fail(f"Python {v.major}.{v.minor} — need 3.10+")
    else:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")


def check_imports() -> None:
    for name in (
        "yaml",
        "pydantic",
        "dotenv",
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "yfinance",
        "bs4",
        "requests",
        "httpx",
        "aiokafka",
        "crewai",
        "pytest",
    ):
        try:
            importlib.import_module(name)
            ok(f"import {name}")
        except Exception as exc:  # noqa: BLE001
            fail(f"import {name}: {exc}")


def check_files() -> None:
    required = [
        ROOT / "requirements.txt",
        ROOT / ".env.example",
        ROOT / "docker-compose.yml",
        ROOT / "Dockerfile",
        ROOT / "deploy" / "searxng" / "settings.yml",
        ROOT / "trading_crew" / "config" / "agents.yaml",
        ROOT / "trading_crew" / "config" / "tasks.yaml",
        ROOT / "web" / "backend" / "app.py",
    ]
    for path in required:
        if path.is_file():
            ok(path.relative_to(ROOT).as_posix())
        else:
            fail(f"missing {path.relative_to(ROOT)}")

    env = ROOT / ".env"
    if env.is_file():
        ok(".env present (not shipped — fill from .env.example on a new machine)")
    else:
        warn(".env missing — copy .env.example and set VLLM_LLM_* (or LOCAL_LLM_*)")


def check_yaml() -> None:
    try:
        import yaml
    except Exception:
        return
    for rel in (
        "trading_crew/config/agents.yaml",
        "trading_crew/config/tasks.yaml",
        "deploy/searxng/settings.yml",
        "docker-compose.yml",
    ):
        path = ROOT / rel
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            ok(f"parse {rel}")
        except Exception as exc:  # noqa: BLE001
            fail(f"parse {rel}: {exc}")


def check_env_example_knobs() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for knob in (
        "VLLM_LLM_BASE_URL",
        "TRADINGCREW_KAFKA_BOOTSTRAP",
        "SEARXNG_URL",
        "GEMINI_WEB_SEARCH",
        "GEMINI_SEARCH_HOME",
        "TRADINGCREW_WEB_PORT",
        "TRADINGCREW_PM_ALIGN_GUARD",
    ):
        if knob in text:
            ok(f".env.example has {knob}")
        else:
            fail(f".env.example missing {knob}")


def check_no_committed_secrets() -> None:
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if not example.is_file():
        return
    ex = example.read_text(encoding="utf-8")
    for needle in ("tvly-", "sk-", "fc-", "52f2e5ba"):
        if needle in ex:
            fail(f".env.example looks like it contains a real secret ({needle!r})")
    if env.is_file() and "VLLM_LLM_API_KEY=" in env.read_text(encoding="utf-8"):
        ok(".env is local-only (gitignored); do not copy it into the zip")


def main() -> int:
    print(f"TradingCrew setup check  ({ROOT})")
    check_python()
    check_files()
    check_imports()
    check_yaml()
    check_env_example_knobs()
    check_no_committed_secrets()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
        return 1
    print(f"all required checks passed  ({len(WARNINGS)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
