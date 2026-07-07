<div align="center">

# TradingCrew

### Multi-agent **stock + commodity** research, with a deterministic risk engine after the LLM.

A team of specialist LLM agents debates a trade idea; the verdict then runs through auditable, code-only modules for sizing, risk gates, execution simulation, and backtesting — so the *narrative* is the LLM's, but the *numbers* are pure code.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![Built with CrewAI](https://img.shields.io/badge/Built%20with-CrewAI-FF5A50.svg)](https://docs.crewai.com)
[![Docs](https://img.shields.io/badge/docs-full%20reference-8A2BE2.svg)](docs/README.md)

![Workflow diagram](docs/images/workflow.png)

</div>

---

## Features

- **Two dashboards, one backend** — equities (`/stock`) and futures (`/commodity`, with curve / COT / seasonality) share the same deterministic core.
- **Pick your model in the UI** — hosted vLLM, local vLLM, OpenAI, or Anthropic, switchable per run.
- **Multi-region tickers** — `AAPL`, `RELIANCE.NS`, `0700.HK`, `7203.T`, `HSBA.L`, `CL=F`, … auto-resolved with the right macro basket + news themes.
- **India-first data path** — `.NS` / `.BO` tickers can route through **screener.in** + **upstox v3** for clean fundamentals, shareholding, and OHLCV.
- **Auditable risk layer** — every decision passes a typed pipeline: bridge → simulator → episodic memory → reflective critic → Kelly/vol/CVaR sizer + hard risk gates → walk-forward backtest → allocator.
- **Refresh-safe background sessions** — queue multiple tickers; runs are owned by the server, not the browser tab.
- **Learns from outcomes** — reflection on resolved trades, walk-forward grid search, PPO on the simulator, and RLMF (prompt-only or LoRA fine-tune).

## Screenshots

|  |  |
| --- | --- |
| **Stock dashboard** | **Commodity dashboard** |
| ![Stock dashboard](docs/images/dashboard-stock.png) | ![Commodity dashboard](docs/images/dashboard-commodity.png) |
| **RLMF — fine-tune + serve** | **RLMF — training episodes** |
| ![RLMF fine-tune and serve](docs/images/rlmf-finetune-serve.png) | ![RLMF episodes](docs/images/rlmf-episodes.png) |

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env       # set VLLM_LLM_* + TAVILY_API_KEY at minimum

./run_web.sh               # serves on http://localhost:8001
```

Open <http://localhost:8001>, pick a ticker, and hit **Run analysis**. For a headless run: `.venv/bin/python main.py <TICKER>`.

## Documentation

**➡️ [Full documentation](docs/README.md)** — configuration, model/serving guide, the deterministic design, and a [finance primer](docs/finance/README.md) for non-finance readers.

## Inspiration & references

- **TauricResearch/TradingAgents** — the high-level multi-agent debate topology (analyst team → bull/bear → trader → 3-way risk → PM) is inspired by this Apache-2.0 project. <https://github.com/TauricResearch/TradingAgents>
- **Xiao et al., "TradingAgents: Multi-Agents LLM Financial Trading Framework"** — arXiv:[2412.20138](https://arxiv.org/abs/2412.20138) (2025).
- **Deterministic Layer-A design** — the M1–M7 contract (bridge, simulator, memory, critic, sizer + gates, walk-forward, allocator) draws on arXiv:[2605.19337](https://arxiv.org/pdf/2605.19337).

TradingCrew is an independent implementation built on **CrewAI**, with its own futures dashboard, FastAPI/WebSocket web UI, and deterministic post-LLM pipeline. No source code is copied from the upstream project. See [`NOTICE`](NOTICE) for the full attribution.

## Disclaimer

For **research and educational purposes only**. Market data is delayed and back-tested performance is not a guarantee of future returns. **Nothing here is investment advice.**

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
