"""Tests for ``web.backend.logging_setup``.

The module is the single place where the web server picks up its
stdout + rotating-file logging.  Once installed it has to be:

* Idempotent — uvicorn-reload imports ``app.py`` repeatedly in the same
  Python process; doubling the handler list every time would cause each
  log line to be duplicated in the file and on stdout.
* Picked up by both the root logger and the uvicorn lifecycle loggers
  (``uvicorn``, ``uvicorn.access``, ``uvicorn.error``) so request logs
  show up in the file.
* Honour the ``TRADINGCREW_LOG_DIR`` env knob.
"""

from __future__ import annotations

import logging
import logging.handlers

import pytest

from web.backend import logging_setup


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Each test starts with a clean root logger + module flag."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_flag = logging_setup._configured  # noqa: SLF001 — test fixture

    yield

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    logging_setup._configured = saved_flag  # noqa: SLF001 — test fixture


def test_configure_logging_attaches_stdout_and_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("TRADINGCREW_LOG_LEVEL", raising=False)
    logging_setup._configured = False  # noqa: SLF001 — test fixture

    log_path = logging_setup.configure_logging()

    root = logging.getLogger()
    handler_types = {type(h) for h in root.handlers}
    assert logging.handlers.RotatingFileHandler in handler_types, (
        f"expected a RotatingFileHandler, got {handler_types}"
    )
    assert logging.StreamHandler in handler_types, (
        f"expected a stdout StreamHandler, got {handler_types}"
    )
    assert log_path.parent == tmp_path
    assert log_path.name == "web.log"


def test_log_records_actually_hit_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("TRADINGCREW_LOG_LEVEL", raising=False)
    logging_setup._configured = False  # noqa: SLF001 — test fixture

    log_path = logging_setup.configure_logging()
    logging.getLogger("tradingcrew.test").info("sentinel-message-12345")
    for handler in logging.getLogger().handlers:
        handler.flush()

    contents = log_path.read_text()
    assert "sentinel-message-12345" in contents
    assert "tradingcrew.test" in contents


def test_idempotent_within_process(monkeypatch, tmp_path):
    """A second call must not double-stack handlers — that would emit
    every log line twice into the rotating file."""
    monkeypatch.setenv("TRADINGCREW_LOG_DIR", str(tmp_path))
    logging_setup._configured = False  # noqa: SLF001 — test fixture

    logging_setup.configure_logging()
    first_handlers = list(logging.getLogger().handlers)

    logging_setup.configure_logging()
    second_handlers = list(logging.getLogger().handlers)

    assert len(first_handlers) == len(second_handlers), (
        f"second call doubled handlers: {first_handlers} -> {second_handlers}"
    )


def test_uvicorn_loggers_propagate_to_root(monkeypatch, tmp_path):
    """Uvicorn's default handlers must be cleared and propagate enabled
    so their messages reach our root handlers (and the log file)."""
    monkeypatch.setenv("TRADINGCREW_LOG_DIR", str(tmp_path))
    logging_setup._configured = False  # noqa: SLF001 — test fixture

    # Simulate uvicorn pre-installing a stdout handler.
    uv = logging.getLogger("uvicorn.access")
    uv.handlers = [logging.StreamHandler()]
    uv.propagate = False

    log_path = logging_setup.configure_logging()
    assert uv.propagate is True
    assert uv.handlers == []

    logging.getLogger("uvicorn.access").info("uvicorn-sentinel-67890")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "uvicorn-sentinel-67890" in log_path.read_text()
