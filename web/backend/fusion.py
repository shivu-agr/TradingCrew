"""Shared signal-fusion input assembly.

Gathers the fusion views for a ticker — the LLM decision (latest logged
proposal), the L3 tuned expectancy edge, and the optional live
crowd-positioning signal — each fail-soft. L4 PPO is not assembled
(removed from the live prediction path). Lives in its own module so both
the HTTP endpoint *and* the workflow runner's optional in-run fusion
step can reuse it without a circular import.
"""

from __future__ import annotations

from datetime import datetime

from .charts import _fetch_ohlcv


def assemble_fusion_inputs(ticker: str, include_crowd: bool, expectancy_source: str = "pso"):
    """Gather the available signals for ``ticker`` (each source fail-soft).

    Returns ``(inputs, sources, error)`` where ``inputs`` is a dict with keys
    ``llm``/``policy``/``expectancy``/``crowd`` (any of policy/expectancy/crowd
    may be ``None``), ``sources`` flags which were populated, and ``error`` is a
    human note when nothing usable exists (no logged decision for the ticker).
    """
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.memory import EpisodicMemory, OutcomeStatus
    from trading_crew.agentic.reflection import _default_memory_path
    from trading_crew.agentic.signals import (
        ExpectancySignal,
        LLMSignal,
        pso_tuned_expectancy,
    )

    sources = {"llm": False, "policy": False, "expectancy": False, "crowd": False}

    # --- LLM: latest logged proposal --------------------------------------
    mem = EpisodicMemory(_default_memory_path())
    eps = [e for e in mem.all_episodes() if e.symbol.upper() == ticker]
    if not eps:
        return None, sources, "No logged decisions for this ticker yet."
    eps.sort(key=lambda e: e.decision_ts)
    latest = eps[-1]
    prop = latest.action_proposal or {}
    llm = LLMSignal(
        side=str(prop.get("side") or "ABSTAIN").upper(),
        conviction=float(prop.get("conviction_score") or 0.0),
        expected_return_pct=float(prop.get("expected_return_pct") or 0.0),
        target_weight=prop.get("target_weight"),
    )
    sources["llm"] = True

    # L4 PPO is not on the live prediction path (OHLCV-only advisory,
    # no Resolve/desk evidence). Fusion no longer loads a policy.
    policy = None

    # --- L3 tuned expectancy ----------------------------------------------
    # Preferred source is the PSO-tuned M5 config's per-trade expectancy
    # (the swarm optimizer searches for the config that maximises the
    # risk-adjusted objective — a stronger edge estimate than a raw mean).
    # ``proxy`` uses the mean realised/alpha of resolved episodes; ``none``
    # disables the gate.
    expectancy = None
    if expectancy_source == "pso":
        try:
            proposals = []
            for e in eps:
                try:
                    proposals.append(ActionProposal(**e.action_proposal))
                except Exception:  # noqa: BLE001
                    continue
            if proposals:
                end_dt = datetime.utcnow()
                ohlcv = _fetch_ohlcv(ticker, end_dt, 365 * 5)
                if ohlcv is not None and not ohlcv.empty:
                    expectancy = pso_tuned_expectancy(proposals, {ticker: ohlcv})
                    if expectancy is not None:
                        sources["expectancy"] = True
        except Exception:  # noqa: BLE001 — expectancy is an optional gate
            expectancy = None
    elif expectancy_source == "proxy":
        resolved = [
            e
            for e in eps
            if e.outcome_status == OutcomeStatus.RESOLVED and e.realised_return is not None
        ]
        if resolved:
            edges = [
                float(e.alpha_return if e.alpha_return is not None else e.realised_return)
                for e in resolved
            ]
            wins = sum(1 for e in edges if e > 0)
            expectancy = ExpectancySignal(
                expectancy=sum(edges) / len(edges),
                win_rate=wins / len(edges),
                n_samples=len(edges),
            )
            sources["expectancy"] = True

    # --- Crowd (optional; live tool call) ---------------------------------
    crowd = None
    if include_crowd:
        try:
            from trading_crew.agentic.swarm.crowd import compute_crowd_signal
            from trading_crew.agentic.swarm.crowd_fit import load_crowd_config
            from trading_crew.tools import _gather_crowd_inputs

            crowd_inputs, _notes = _gather_crowd_inputs(ticker)
            sig = compute_crowd_signal(crowd_inputs, load_crowd_config())
            if sig.n_components > 0:
                crowd = sig
                sources["crowd"] = True
        except Exception:  # noqa: BLE001 — crowd is optional
            crowd = None

    return {"llm": llm, "policy": policy, "expectancy": expectancy, "crowd": crowd}, sources, None
