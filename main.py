"""CLI entry point: ``python main.py NTNX --debate-rounds 2 --risk-rounds 1``"""

from __future__ import annotations

import argparse
import json
import sys

from trading_crew import TradingCrew
from trading_crew.data_sources import is_indian_ticker
from trading_crew.graph import run_stock_graph, use_stock_graph


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the TradingCrew v3 stock workflow.")
    p.add_argument("ticker", nargs="?", default="NTNX", help="Stock ticker (default: NTNX)")
    p.add_argument("--debate-rounds", type=int, default=2,
                   help="Bull/Bear debate rounds (default: 2)")
    p.add_argument("--risk-rounds", type=int, default=1,
                   help="Risk-team debate rounds (default: 1)")
    p.add_argument(
        "--mode",
        choices=["autonomous", "deterministic", "harness"],
        default="autonomous",
        help="autonomous = LangGraph; deterministic = sequential CrewAI; harness = observe/dispatch loop",
    )
    p.add_argument("--quality-review", action="store_true",
                   help="Run the advisory Quality Reviewer before the PM")
    p.add_argument("--compliance", action="store_true",
                   help="Run the advisory Compliance Officer before the PM")
    p.add_argument("--no-memory", action="store_true",
                   help="Disable shared memory across agents")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    graph_mode = use_stock_graph("stock", args.mode)
    harness_mode = args.mode == "harness"
    label = (
        "harness loop" if harness_mode else ("LangGraph" if graph_mode else "sequential CrewAI")
    )
    print(f"=== TradingCrew v3 ({label}) ===")
    print(f"  Ticker:         {args.ticker}")
    print(f"  Run mode:       {args.mode}")
    if not harness_mode:
        print(f"  Debate rounds:  {args.debate_rounds}")
        print(f"  Risk rounds:    {args.risk_rounds}")
    print(f"  Quality review: {'on' if args.quality_review else 'off'}")
    print(f"  Compliance:     {'on' if args.compliance else 'off'}")
    print(f"  Memory:         {'off' if harness_mode or args.no_memory else 'on'}")
    print()

    tc = TradingCrew(
        ticker=args.ticker,
        debate_rounds=args.debate_rounds,
        risk_rounds=args.risk_rounds,
        memory=False if harness_mode else not args.no_memory,
        quality_review_enabled=args.quality_review,
        compliance_enabled=args.compliance,
        run_mode=args.mode,
    )

    if harness_mode:
        from trading_crew.harness import run_harness

        agents = tc.materialize_agents()
        print(f"Agents: {len(agents)} (harness-spawned)\n")
        result = run_harness(tc, agents)
    elif graph_mode:
        agents = tc.materialize_agents()
        print(f"Agents: {len(agents)} (graph-scheduled)\n")
        market = "IN" if is_indian_ticker(args.ticker) else "US"
        result = run_stock_graph(
            tc,
            agents,
            ui_debate_rounds=args.debate_rounds,
            ui_risk_rounds=args.risk_rounds,
            quality_review_enabled=args.quality_review,
            compliance_enabled=args.compliance,
            market=market,
        )
    else:
        crew = tc.crew()
        print(f"Agents: {len(crew.agents)} (sequential)\n")
        result = crew.kickoff(inputs={"ticker": args.ticker})

    print()
    print("=" * 70)
    print(f"FINAL DECISION for {args.ticker}")
    print("=" * 70)
    if result.pydantic is not None:
        print(json.dumps(result.pydantic.model_dump(), indent=2))
    else:
        print(result.raw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
