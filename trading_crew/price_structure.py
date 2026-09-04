"""Deterministic price-structure signals — a Firefly-style card lifted from
OHLCV alone (no LLM, fully citable).

Motivation & evidence
----------------------
A 6-year, 198-name, 22k-sample point-in-time probe (``scripts/
price_structure_probe.py``) measured which price-derived signals actually
condition forward outcomes at the workflow's medium-term horizon:

* **Trend/momentum direction has NO edge** on this liquid universe — Hurst
  buckets all show ~0.50 60d hit-rate.  So Hurst is emitted as *descriptive
  context only*, never a directional vote.
* **Realised-vol percentile predicts forward vol/drawdown** (clean, monotonic:
  low/mid/high vol-pct -> 0.264/0.285/0.304 forward 60d vol).  -> a *sizing*
  signal.
* **BOCPD regime age**: right after a detected break, forward vol and move
  size rise and momentum reliability dips -> a mild "don't chase / size down"
  signal.
* **Dip + oversold is a real mean-reversion edge**: a sharp 20d drop (<-8%)
  while in the bottom quartile of the 52w range returned +7.5% fwd60 (66%
  win) vs +3.6% (58%) baseline — the "buy a temporary sentiment dip" setup.

Design
------
The card reuses :class:`trading_crew.report_signals.ReportSignal` and the
canonical ``SIGNAL | NAME | DIRECTION | conf=X | rationale [source]`` grammar,
so the SAME parser (:func:`report_signals.parse_signal_card`) and PM-ledger
plumbing already wired for the reports desk lift these into structure.  The
signals are risk/entry context — they modulate *size* (via
:meth:`PriceStructure.risk_mult`) and *conviction magnitude* (via the PM
ledger), never dictate a side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from trading_crew.report_signals import ReportSignal, render_signal_line

# Thresholds (probe-derived; explicit, no hidden policy).
_DIP_RET20 = -0.08          # a "sharp dip" = 20d return below -8%
_OVERSOLD_POS = 0.25        # bottom quartile of the trailing 52w range
_YOUNG_REGIME_DAYS = 45.0   # BOCPD regime age below this = fresh break
_VOL_ELEVATED = 0.80        # realised-vol percentile at/above = elevated
_VOL_SUBDUED = 0.33


# ---------------------------------------------------------------------------
# Point-in-time signal math (mirrors the validated probe)
# ---------------------------------------------------------------------------

def dfa_alpha(ret: np.ndarray) -> float:
    """Detrended Fluctuation Analysis exponent (~0.5 random, >0.5 trending,
    <0.5 mean-reverting).  NaN when the window is too short."""
    ret = np.asarray(ret, float)
    ret = ret[np.isfinite(ret)]
    n = ret.size
    if n < 64:
        return float("nan")
    y = np.cumsum(ret - ret.mean())
    smax = n // 4
    if smax < 9:
        return float("nan")
    scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(smax), 12)).astype(int))
    scales = scales[scales >= 8]
    F, used = [], []
    for s in scales:
        nseg = n // s
        if nseg < 2:
            continue
        t = np.arange(s)
        rms = [np.sqrt(np.mean((y[v * s:(v + 1) * s]
               - np.polyval(np.polyfit(t, y[v * s:(v + 1) * s], 1), t)) ** 2))
               for v in range(nseg)]
        F.append(np.sqrt(np.mean(np.square(rms))))
        used.append(s)
    if len(F) < 3:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(F), 1)[0])


def bocpd_regime_age(logret: np.ndarray, hazard_lambda: float = 250.0,
                     rmax: int = 750) -> float:
    """Expected run length (regime age, in trading days) of the CURRENT regime
    via Bayesian Online Change-Point Detection (Adams & MacKay 2007).

    Normal-Inverse-Gamma prior auto-scaled to the series variance (else the
    prior swamps daily-return moves and the detector never fires).  A *young*
    value means a regime break happened recently.  Returns NaN on degenerate
    input.

    Note: ``P(run_length=0)`` is structurally equal to the hazard in BOCPD, so
    regime age — not an instantaneous "break probability" — is the usable
    signal (confirmed in the probe)."""
    x = np.asarray(logret, float)
    x = x[np.isfinite(x)]
    T = x.size
    if T < 64:
        return float("nan")
    beta0 = max(1e-8, 0.5 * float(np.var(x)))
    mu0, kappa0, alpha0 = 0.0, 1.0, 1.0
    muT = np.array([mu0]); kapT = np.array([kappa0])
    alpT = np.array([alpha0]); betT = np.array([beta0])
    rl = np.array([1.0])
    H = 1.0 / hazard_lambda
    try:
        from scipy.special import gammaln  # type: ignore
    except Exception:  # pragma: no cover
        _lg = np.vectorize(math.lgamma)
        def gammaln(a):  # noqa: E306
            return _lg(np.asarray(a, float))
    exp_rl = float("nan")
    for t in range(T):
        xt = x[t]
        df = 2 * alpT
        scale = np.sqrt(betT * (kapT + 1) / (alpT * kapT))
        z = (xt - muT) / scale
        logc = (gammaln((df + 1) / 2) - gammaln(df / 2)
                - 0.5 * np.log(df * math.pi) - np.log(scale))
        pred = np.exp(logc - (df + 1) / 2 * np.log1p(z * z / df))
        growth = rl * pred * (1 - H)
        cp = float(np.sum(rl * pred * H))
        new = np.concatenate([[cp], growth])
        s = new.sum()
        if not np.isfinite(s) or s <= 0:
            new = np.concatenate([[1.0], np.zeros_like(growth)]); s = 1.0
        new /= s
        # NIG posterior for the existing run lengths, computed from the OLD
        # arrays, then the fresh-regime prior is prepended for r=0.
        mu_n = (kapT * muT + xt) / (kapT + 1)
        bet_n = betT + (kapT * (xt - muT) ** 2) / (2 * (kapT + 1))
        muT = np.concatenate([[mu0], mu_n])
        kapT = np.concatenate([[kappa0], kapT + 1])
        alpT = np.concatenate([[alpha0], alpT + 0.5])
        betT = np.concatenate([[beta0], bet_n])
        rl = new
        if rl.size > rmax:  # fold overflow tail into last bin to bound cost
            head = rl[:rmax].copy(); head[-1] += rl[rmax:].sum()
            rl = head
            muT, kapT, alpT, betT = muT[:rmax], kapT[:rmax], alpT[:rmax], betT[:rmax]
        exp_rl = float(np.dot(np.arange(rl.size), rl))
    return exp_rl


# ---------------------------------------------------------------------------
# The structured card
# ---------------------------------------------------------------------------

_CARD_HEADER = "## PRICE STRUCTURE SIGNALS"


@dataclass
class PriceStructure:
    """Deterministic price-structure read + the signals lifted from it."""

    ann_vol: float
    vol_pct: float
    atr_pct: float
    pos_in_range: float
    drawdown_from_high: float
    regime_age_days: float
    hurst: float
    ret20: float
    dip_buy: bool
    signals: List[ReportSignal] = field(default_factory=list)
    source: str = "yfinance daily OHLCV (computed)"

    # --- size modulation (defensive only — never amplifies) ---------------
    def risk_mult(self, floor: float = 0.5) -> Tuple[float, str]:
        """Vol-state / young-regime penalty in ``[floor, 1.0]``.

        Mirrors :func:`sizing.debate_to_risk_mult`: multiplicative penalties
        that can only *reduce* size.  High realised-vol percentile and a fresh
        regime break each shave size — the probe showed both predict higher
        forward vol / drawdown.  The DIP_BUY setup is directional and handled
        by PM conviction, NOT here, so it never inflates size."""
        mult = 1.0
        parts: List[str] = []
        if np.isfinite(self.vol_pct):
            if self.vol_pct >= _VOL_ELEVATED:
                mult *= 0.8; parts.append(f"vol pct {self.vol_pct:.0%} (elevated) ×0.8")
            elif self.vol_pct >= 0.66:
                mult *= 0.9; parts.append(f"vol pct {self.vol_pct:.0%} (high) ×0.9")
        if np.isfinite(self.regime_age_days) and self.regime_age_days < _YOUNG_REGIME_DAYS:
            mult *= 0.9; parts.append(f"fresh regime ({self.regime_age_days:.0f}d) ×0.9")
        mult = max(floor, min(1.0, mult))
        return mult, (" · ".join(parts) if parts else "no vol-state size penalty")

    def render_card(self) -> str:
        """Appendable ``## PRICE STRUCTURE SIGNALS`` block for the analyst tool."""
        if not self.signals:
            return ""
        return "\n".join([_CARD_HEADER] + [render_signal_line(s) for s in self.signals])

    def render_pm_ledger(self) -> str:
        """Authoritative, non-directive PM block (completeness guarantee).

        Guarantees the parsed price-structure read reaches the final decision
        even if the analyst prose drops the card.  It prescribes no side — it
        forces the vol-state (size) and any DIP_BUY setup to be weighed and
        cited."""
        decisive = [s for s in self.signals
                    if s.direction and s.direction.upper() not in ("INSUFFICIENT_DATA", "NONE")]
        if not decisive:
            return ""
        lines = [
            "## PRICE-STRUCTURE DESK — MACHINE-COMPUTED SIGNALS (authoritative; do not ignore)",
            "Computed deterministically from OHLCV, independent of any analyst "
            "summary. These are RISK/ENTRY context — they inform SIZE and "
            "CONVICTION MAGNITUDE, not direction:",
        ]
        lines += [render_signal_line(s) for s in decisive]
        rm, why = self.risk_mult()
        if rm < 1.0:
            lines.append(f"SIZE_GUIDANCE: vol-state suggests trimming size (risk multiplier {rm:.2f} — {why}).")
        if self.dip_buy:
            lines.append(
                "DIP_BUY setup present: a sharp recent drop while oversold. "
                "Historically (6y/198-name probe) this cohort returned ~+7.5% "
                "over 60d with a 66% win rate vs a +3.6%/58% baseline — a "
                "mean-reversion ENTRY edge. If your thesis is BUY, this "
                "supports higher conviction; if SELL, name the catalyst that "
                "overrides the base-rate bounce.")
            lines.append(
                "You MUST weigh the vol-state in sizing and cite the DIP_BUY "
                "read. (Completeness check — it does not prescribe a side.)")
        else:
            lines.append(
                "You MUST weigh the vol-state in sizing. (Completeness check "
                "— it does not prescribe a side.)")
        return "\n".join(lines)


def _vol_percentile(logret: np.ndarray, win: int = 20, lookback: int = 252) -> Tuple[float, float]:
    r = np.asarray(logret, float)
    r = r[np.isfinite(r)]
    if r.size < win + 20:
        return float("nan"), float("nan")
    roll = pd.Series(r).rolling(win).std().to_numpy() * math.sqrt(252)
    cur = roll[-1]
    hist = roll[-lookback:] if roll.size >= lookback else roll
    hist = hist[np.isfinite(hist)]
    if not np.isfinite(cur) or hist.size < 20:
        return float(cur), float("nan")
    return float(cur), float((hist <= cur).mean())


def _atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, win: int = 14) -> float:
    h, l, c = (np.asarray(a, float) for a in (high, low, close))
    if c.size < win + 1:
        return float("nan")
    prev = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev), np.abs(l[1:] - prev)))
    atr = pd.Series(tr).rolling(win).mean().to_numpy()[-1]
    return float(atr / c[-1]) if c[-1] else float("nan")


def _position_in_range(close: np.ndarray, win: int = 252) -> Tuple[float, float]:
    c = np.asarray(close, float)
    c = c[np.isfinite(c)]
    if c.size < 20:
        return float("nan"), float("nan")
    w = c[-win:] if c.size >= win else c
    lo, hi, cur = w.min(), w.max(), c[-1]
    pos = float((cur - lo) / (hi - lo)) if hi > lo else float("nan")
    dd = float(cur / hi - 1.0) if hi else float("nan")
    return pos, dd


def compute_price_structure(df: pd.DataFrame, source: str = "yfinance daily OHLCV (computed)") -> Optional[PriceStructure]:
    """Build the :class:`PriceStructure` card from a High/Low/Close frame.

    ``df`` must be chronological (oldest first) with ``Close`` (High/Low
    optional — ATR is skipped if absent).  Returns ``None`` when there isn't
    enough history to compute anything decisive (never fabricates)."""
    if df is None or df.empty or "Close" not in df:
        return None
    close = df["Close"].to_numpy(float)
    close = close[np.isfinite(close)]
    if close.size < 80:
        return None
    high = df["High"].to_numpy(float) if "High" in df else close
    low = df["Low"].to_numpy(float) if "Low" in df else close
    logret = np.diff(np.log(close), prepend=np.log(close[0]))

    ann_vol, vol_pct = _vol_percentile(logret)
    atr = _atr_pct(high, low, close)
    pos, dd = _position_in_range(close)
    regime_age = bocpd_regime_age(logret)
    hurst = dfa_alpha(logret[-252:]) if close.size >= 256 else dfa_alpha(logret)
    ret20 = float(close[-1] / close[-21] - 1.0) if close.size >= 21 else float("nan")
    dip_buy = bool(np.isfinite(ret20) and ret20 < _DIP_RET20
                   and np.isfinite(pos) and pos < _OVERSOLD_POS)

    sigs: List[ReportSignal] = []

    # VOL_STATE — the validated sizing signal.  ELEVATED threshold matches the
    # 0.66 percentile where risk_mult() starts trimming size, so the card label
    # and the size penalty agree.
    if np.isfinite(vol_pct):
        if vol_pct >= 0.66:
            vdir, vconf = "ELEVATED", 0.8 if vol_pct >= _VOL_ELEVATED else 0.6
        elif vol_pct <= _VOL_SUBDUED:
            vdir, vconf = "SUBDUED", 0.6
        else:
            vdir, vconf = "NORMAL", 0.5
        sigs.append(ReportSignal(
            name="VOL_STATE", direction=vdir, confidence=vconf,
            rationale=f"annualised realised vol {ann_vol:.0%}, {vol_pct:.0%} percentile vs trailing year",
            source=source, computed=True))

    # PRICE_POSITION — where in the 52w range (context for the dip read).
    if np.isfinite(pos):
        if pos >= 0.8:
            pdir = "NEAR_HIGH"
        elif pos <= 0.25:
            pdir = "NEAR_LOW"
        else:
            pdir = "MID_RANGE"
        sigs.append(ReportSignal(
            name="PRICE_POSITION", direction=pdir, confidence=0.6,
            rationale=f"{pos:.0%} of 52w range, {dd:+.0%} from 52w high"
                      + (f", ATR {atr:.1%}" if np.isfinite(atr) else ""),
            source=source, computed=True))

    # REGIME_AGE — BOCPD "recently broke?" flag.
    if np.isfinite(regime_age):
        rdir = "FRESH_BREAK" if regime_age < _YOUNG_REGIME_DAYS else "ESTABLISHED"
        sigs.append(ReportSignal(
            name="REGIME_AGE", direction=rdir, confidence=0.5,
            rationale=f"BOCPD expected regime age {regime_age:.0f} trading days",
            source=source, computed=True))

    # DIP_BUY — the measured mean-reversion entry edge.
    sigs.append(ReportSignal(
        name="DIP_BUY", direction="SETUP" if dip_buy else "NONE",
        confidence=0.66 if dip_buy else 0.0,
        rationale=(f"20d return {ret20:+.0%} while {pos:.0%} of 52w range "
                   "(oversold dip — 66% 60d win-rate in probe)") if dip_buy
                  else "no sharp-dip / oversold setup",
        source=source, computed=True))

    # HURST — DESCRIPTIVE ONLY (probe: no directional edge on this universe).
    if np.isfinite(hurst):
        if hurst > 0.55:
            hdir = "TRENDING"
        elif hurst < 0.45:
            hdir = "MEAN_REVERTING"
        else:
            hdir = "RANDOM_WALK"
        sigs.append(ReportSignal(
            name="HURST", direction=hdir, confidence=0.3,
            rationale=f"DFA exponent {hurst:.2f} (context only — not a directional signal)",
            source=source, computed=True))

    return PriceStructure(
        ann_vol=ann_vol, vol_pct=vol_pct, atr_pct=atr, pos_in_range=pos,
        drawdown_from_high=dd, regime_age_days=regime_age, hurst=hurst,
        ret20=ret20, dip_buy=dip_buy, signals=sigs, source=source)
