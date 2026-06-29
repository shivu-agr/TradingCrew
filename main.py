"""CLI entry point: ``python main.py NTNX --debate-rounds 2 --risk-rounds 1``"""

from __future__ import annotations

import argparse
import json
import sys

from trading_crew import TradingCrew


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the TradingCrew workflow on a ticker.")
    p.add_argument("ticker", nargs="?", default="NTNX", help="Stock ticker (default: NTNX)")
    p.add_argument("--debate-rounds", type=int, default=2,
                   help="Bull/Bear debate rounds (default: 2)")
    p.add_argument("--risk-rounds", type=int, default=1,
                   help="Risk-team debate rounds (default: 1)")
    p.add_argument("--no-memory", action="store_true",
                   help="Disable shared memory across agents")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"=== TradingCrew ===")
    print(f"  Ticker:         {args.ticker}")
    print(f"  Debate rounds:  {args.debate_rounds}")
    print(f"  Risk rounds:    {args.risk_rounds}")
    print(f"  Memory:         {'off' if args.no_memory else 'on'}")
    print()

    tc = TradingCrew(
        ticker=args.ticker,
        debate_rounds=args.debate_rounds,
        risk_rounds=args.risk_rounds,
        memory=not args.no_memory,
    )
    crew = tc.crew()

    print(f"Agents: {len(crew.agents)}, Tasks: {len(crew.tasks)}\n")

    result = crew.kickoff(inputs={"ticker": args.ticker})

    print()
    print("=" * 70)
    print(f"FINAL DECISION for {args.ticker}")
    print("=" * 70)
    if result.pydantic is not None:
        print(json.dumps(result.pydantic.model_dump(), indent=2))
    else:
        print(result.raw)

    print()
    print("Usage:", crew.usage_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
