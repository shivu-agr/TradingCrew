#!/usr/bin/env python3
"""Offline probe: do deterministic price-structure signals add value?

This is a *research prototype*, NOT wired into the crew.  It answers two
questions raised in review before we commit any workflow changes:

  1. EFFICACY (pure price, multi-year backtest) — at the workflow's
     medium-term horizon (20/60/120 trading days), do these signals
     actually condition forward returns / risk?
       - Hurst/DFA persistence   -> is a directional (momentum) thesis
                                     more reliable in trending vs
                                     mean-reverting regimes?
       - Realised-vol percentile -> does it predict forward drawdown
                                     (i.e. is it a sizing signal)?
       - BOCPD change-point       -> right after a detected regime break,
                                     does momentum hit-rate drop / drawdown
                                     rise ("don't chase") ?

  2. WORKFLOW OVERLAY (real decisions in prediction.db) — the crew flips
     BUY<->SELL on the *same name* across consecutive days for a 180d
     thesis.  Do those flips concentrate in low-Hurst (whipsaw) /
     high-instability (near change-point) regimes?  If so, a Hurst/BOCPD
     conviction gate would have stabilised the book.

All signals are computed **point-in-time** (only data up to the decision
date) so there is no look-ahead.  Outcomes for the live episodes have not
resolved yet (180d horizon, runs are days old), which is exactly why the
efficacy question is answered on historical price data instead.

Usage:
    ./.venv/bin/python scripts/price_structure_probe.py            # sample universe
    ./.venv/bin/python scripts/price_structure_probe.py --full     # all 201 names
    ./.venv/bin/python scripts/price_structure_probe.py --years 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.special import gammaln  # type: ignore
except Exception:  # pragma: no cover - scipy optional
    _lg = np.vectorize(math.lgamma)
    def gammaln(a):  # noqa: E302
        return _lg(np.asarray(a, dtype=float))

DB = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
DB_PATH = str(Path(DB) / "prediction.db")
CACHE = Path(__file__).with_name(".cache_prices.pkl")
OUT = Path(__file__).with_name("price_structure_probe_out.json")


# ===========================================================================
# Signal library — all point-in-time, no look-ahead
# ===========================================================================

def dfa_alpha(ret: np.ndarray) -> float:
    """Detrended Fluctuation Analysis exponent on a return series.

    alpha ~ 0.5  random walk (no memory)
    alpha > 0.5  persistent / trending (momentum continues)
    alpha < 0.5  anti-persistent / mean-reverting (moves reverse -> whipsaw)

    DFA is the estimator the literature (East-Asian regime study, 2025)
    found more robust than classic R/S for separating profitable
    trend-following regimes.  Returns NaN when the window is too short.
    """
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
        rms = []
        for v in range(nseg):
            seg = y[v * s:(v + 1) * s]
            coef = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2)))
        F.append(np.sqrt(np.mean(np.square(rms))))
        used.append(s)
    if len(F) < 3:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(F), 1)[0])


def vol_percentile(logret: np.ndarray, win: int = 20, lookback: int = 252) -> Tuple[float, float]:
    """Current annualised realised vol + its percentile vs the trailing year.

    Returns (ann_vol, percentile in [0,1]).  High percentile = the name is
    unusually volatile *for itself* right now -> a sizing / staged-target
    signal, not a direction signal.
    """
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
    pct = float((hist <= cur).mean())
    return float(cur), pct


def atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, win: int = 14) -> float:
    """Average True Range as a fraction of price (a vol / stop-distance unit)."""
    h, l, c = map(lambda a: np.asarray(a, float), (high, low, close))
    if c.size < win + 1:
        return float("nan")
    prev = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev), np.abs(l[1:] - prev)))
    atr = pd.Series(tr).rolling(win).mean().to_numpy()[-1]
    return float(atr / c[-1]) if c[-1] else float("nan")


def position_in_range(close: np.ndarray, win: int = 252) -> Tuple[float, float]:
    """(position-in-52w-range in [0,1], drawdown-from-52w-high as negative %).

    position 1.0 = at the 52w high, 0.0 = at the 52w low.
    """
    c = np.asarray(close, float)
    c = c[np.isfinite(c)]
    if c.size < 20:
        return float("nan"), float("nan")
    w = c[-win:] if c.size >= win else c
    lo, hi, cur = w.min(), w.max(), c[-1]
    pos = float((cur - lo) / (hi - lo)) if hi > lo else float("nan")
    dd = float(cur / hi - 1.0) if hi else float("nan")
    return pos, dd


def _student_t_pdf(x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - loc) / scale
    logc = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * math.pi) - np.log(scale)
    return np.exp(logc - (df + 1) / 2 * np.log1p(z * z / df))


def bocpd(x: np.ndarray, hazard_lambda: float = 250.0,
          mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
          beta0: Optional[float] = None, rmax: int = 750) -> Tuple[np.ndarray, np.ndarray]:
    """Bayesian Online Change-Point Detection (Adams & MacKay 2007).

    Univariate Normal with Normal-Inverse-Gamma prior (Student-t
    predictive) + constant hazard 1/lambda.  Returns, aligned to each
    observation t:
      - cp_prob[t]  : P(run length resets to 0 at t | data<=t) — the
                      instantaneous "a regime just broke" probability.
      - exp_rl[t]   : E[run length] — the regime's *age* in days (a low
                      value means we're early in a fresh, unstable regime).

    Input ``x`` should be a stationary-ish series — we feed daily log
    returns.  This is a risk/regime-stability signal, not a forecaster.
    """
    x = np.asarray(x, float)
    T = x.size
    if beta0 is None:
        # Scale the Normal-Inverse-Gamma prior to the series' own variance so
        # the r=0 (fresh-regime) predictive has a realistic daily-return scale
        # (~1-2%).  Without this the prior variance dwarfs real moves and the
        # detector never fires.  scale0 = sqrt(beta0*(kappa0+1)/(alpha0*kappa0)).
        v = np.nanvar(x[np.isfinite(x)]) if np.isfinite(x).any() else 1e-4
        beta0 = float(max(1e-8, 0.5 * v))
    muT = np.array([mu0]); kapT = np.array([kappa0])
    alpT = np.array([alpha0]); betT = np.array([beta0])
    rl = np.array([1.0])
    H = 1.0 / hazard_lambda
    cp_prob = np.full(T, np.nan)
    exp_rl = np.full(T, np.nan)
    for t in range(T):
        xt = x[t]
        if not np.isfinite(xt):
            if t:
                cp_prob[t], exp_rl[t] = cp_prob[t - 1], exp_rl[t - 1]
            continue
        df = 2 * alpT
        scale = np.sqrt(betT * (kapT + 1) / (alpT * kapT))
        pred = _student_t_pdf(xt, df, muT, scale)
        growth = rl * pred * (1 - H)
        cp = float(np.sum(rl * pred * H))
        new = np.concatenate([[cp], growth])
        s = new.sum()
        if not np.isfinite(s) or s <= 0:
            new = np.concatenate([[1.0], np.zeros_like(growth)])
            s = 1.0
        new /= s
        # NIG posterior update for the *existing* run lengths, then prepend prior
        mu_n = (kapT * muT + xt) / (kapT + 1)
        kap_n = kapT + 1
        alp_n = alpT + 0.5
        bet_n = betT + (kapT * (xt - muT) ** 2) / (2 * (kapT + 1))
        muT = np.concatenate([[mu0], mu_n])
        kapT = np.concatenate([[kappa0], kap_n])
        alpT = np.concatenate([[alpha0], alp_n])
        betT = np.concatenate([[beta0], bet_n])
        rl = new
        if rl.size > rmax:  # fold overflow tail into last kept bin (cost bound)
            head = rl[:rmax].copy(); head[-1] += rl[rmax:].sum()
            rl = head
            muT, kapT, alpT, betT = muT[:rmax], kapT[:rmax], alpT[:rmax], betT[:rmax]
        cp_prob[t] = rl[0]
        exp_rl[t] = float(np.dot(np.arange(rl.size), rl))
    return cp_prob, exp_rl


# ===========================================================================
# Data
# ===========================================================================

def universe_from_db() -> List[str]:
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True)
    rows = c.execute("select distinct ticker from predictions").fetchall()
    c.close()
    return sorted({r[0] for r in rows if r[0]})


def decisions_from_db() -> pd.DataFrame:
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True)
    df = pd.read_sql_query(
        "select ticker, run_date, action, conviction from predictions "
        "where status='done' and action is not null", c)
    c.close()
    df["action"] = df["action"].str.upper()
    return df


def load_prices(tickers: List[str], years: int) -> Dict[str, pd.DataFrame]:
    """Download (and cache) daily OHLC for the universe."""
    cache: Dict[str, pd.DataFrame] = {}
    if CACHE.exists():
        try:
            cache = pickle.loads(CACHE.read_bytes())
        except Exception:
            cache = {}
    missing = [t for t in tickers if t not in cache]
    if missing:
        import yfinance as yf  # lazy import
        print(f"  downloading {len(missing)} tickers ({years}y daily)…", flush=True)
        for i in range(0, len(missing), 40):
            batch = missing[i:i + 40]
            try:
                raw = yf.download(batch, period=f"{years}y", interval="1d",
                                  group_by="ticker", auto_adjust=True, threads=True,
                                  progress=False)
            except Exception as e:
                print(f"    batch {i} failed: {e}", flush=True)
                continue
            for t in batch:
                try:
                    d = raw[t] if len(batch) > 1 else raw
                    d = d.dropna(how="all")
                    if d is not None and not d.empty and "Close" in d:
                        cache[t] = d[["High", "Low", "Close"]].dropna()
                except Exception:
                    pass
            time.sleep(0.5)
        CACHE.write_bytes(pickle.dumps(cache))
    return {t: cache[t] for t in tickers if t in cache and len(cache[t]) > 300}


# ===========================================================================
# Analyses
# ===========================================================================

def hurst_bucket(a: float) -> str:
    if not np.isfinite(a):
        return "n/a"
    if a < 0.45:
        return "mean-revert (H<0.45)"
    if a <= 0.55:
        return "random (0.45-0.55)"
    return "trend (H>0.55)"


def efficacy_backtest(prices: Dict[str, pd.DataFrame], horizons=(20, 60, 120),
                      step=10, hurst_win=252, mom_win=63) -> dict:
    """For sampled decision dates, join PIT signals to forward returns."""
    recs = []
    for t, d in prices.items():
        close = d["Close"].to_numpy(float)
        high = d["High"].to_numpy(float)
        low = d["Low"].to_numpy(float)
        logret = np.diff(np.log(close), prepend=np.log(close[0]))
        cp_prob, exp_rl = bocpd(logret)
        n = close.size
        start = max(hurst_win, mom_win) + 5
        last = n - max(horizons) - 1
        for i in range(start, last, step):
            h = dfa_alpha(logret[i - hurst_win:i])
            _, vpct = vol_percentile(logret[:i])
            atr = atr_pct(high[:i], low[:i], close[:i])
            pos, dd = position_in_range(close[:i])
            mom = 1.0 if close[i] > close[i - mom_win] else -1.0
            ret20 = close[i] / close[i - 20] - 1.0 if i >= 20 else np.nan
            ret5 = close[i] / close[i - 5] - 1.0 if i >= 5 else np.nan
            rec = {"h": h, "vpct": vpct, "atr": atr, "pos": pos, "dd": dd, "mom": mom,
                   "ret20": ret20, "ret5": ret5,
                   "cp": float(cp_prob[i]) if np.isfinite(cp_prob[i]) else np.nan,
                   "rl": float(exp_rl[i]) if np.isfinite(exp_rl[i]) else np.nan}
            for hz in horizons:
                fwd = close[i + hz] / close[i] - 1.0
                rec[f"fwd{hz}"] = fwd
            # forward max drawdown + realised vol over the next 60d
            fwd_path = close[i:i + 60]
            rec["fwd_dd60"] = float(fwd_path.min() / close[i] - 1.0)
            fwd_lr = np.diff(np.log(fwd_path)) if fwd_path.size > 2 else np.array([np.nan])
            rec["fwd_vol60"] = float(np.nanstd(fwd_lr) * math.sqrt(252))
            recs.append(rec)
    df = pd.DataFrame(recs)
    out: dict = {"n_samples": int(len(df)), "n_tickers": len(prices)}

    # 1) Hurst conditioning: momentum hit-rate + return spread by regime
    df["hb"] = df["h"].apply(hurst_bucket)
    hz = 60
    g = {}
    for b, sub in df.groupby("hb"):
        sub = sub.dropna(subset=[f"fwd{hz}"])
        if len(sub) < 30:
            continue
        hit = float((np.sign(sub[f"fwd{hz}"]) == np.sign(sub["mom"])).mean())
        # return of following momentum: mom * fwd
        mom_ret = float((sub["mom"] * sub[f"fwd{hz}"]).mean())
        g[b] = {"n": int(len(sub)),
                "momentum_hit_rate_60d": round(hit, 4),
                "avg_return_following_momentum_60d": round(mom_ret, 5),
                "avg_abs_fwd60": round(float(sub[f"fwd{hz}"].abs().mean()), 5)}
    out["hurst_conditioning"] = g

    # 2) Vol-percentile conditioning: forward 60d drawdown + realised vol
    df["vb"] = pd.cut(df["vpct"], [0, .33, .66, 1.0], labels=["low", "mid", "high"])
    vv = {}
    for b, sub in df.groupby("vb", observed=True):
        sub = sub.dropna(subset=["fwd_dd60"])
        if len(sub) < 30:
            continue
        vv[str(b)] = {"n": int(len(sub)),
                      "avg_fwd_drawdown_60d": round(float(sub["fwd_dd60"].mean()), 5),
                      "p05_fwd_drawdown_60d": round(float(sub["fwd_dd60"].quantile(.05)), 5),
                      "avg_fwd_vol60": round(float(sub["fwd_vol60"].mean()), 5)}
    out["volpct_conditioning"] = vv

    # 3) BOCPD conditioning via regime AGE (expected run length).  Note:
    #    P(run_length=0) is structurally == hazard in BOCPD, so the usable
    #    change-point-proximity signal is a *young* expected run length.
    #    "near change point" = youngest-15% regimes.
    thr = float(df["rl"].quantile(.15))
    out["changepoint_stats"] = {
        "regime_age_days_mean": round(float(df["rl"].mean()), 1),
        "regime_age_days_median": round(float(df["rl"].median()), 1),
        "young_regime_threshold_days": round(thr, 1)}
    df["near_cp"] = df["rl"] < thr
    cc = {}
    for near, sub in df.groupby("near_cp"):
        sub = sub.dropna(subset=[f"fwd{hz}", "fwd_dd60", "fwd_vol60"])
        if len(sub) < 30:
            continue
        hit = float((np.sign(sub[f"fwd{hz}"]) == np.sign(sub["mom"])).mean())
        cc["young_regime_(near_break)" if near else "mature_regime"] = {
            "n": int(len(sub)),
            "momentum_hit_rate_60d": round(hit, 4),
            "avg_fwd_drawdown_60d": round(float(sub["fwd_dd60"].mean()), 5),
            "avg_fwd_vol60": round(float(sub["fwd_vol60"].mean()), 5),
            "avg_abs_fwd60": round(float(sub[f"fwd{hz}"].abs().mean()), 5)}
    out["changepoint_conditioning"] = cc

    # 4) "Buy-low after a sentiment dip" — does a sharp recent drop recover?
    #    Tests whether a temporary down-move (esp. right after a regime break
    #    and while oversold) is followed by a bounce, vs. being the start of a
    #    genuine down-trend.  This is the signal the user wants for entries.
    med_age = float(df["rl"].median())

    def _cohort(mask, label):
        sub = df[mask].dropna(subset=["fwd60", "fwd120"])
        if len(sub) < 30:
            return None
        return {
            "n": int(len(sub)),
            "avg_fwd60": round(float(sub["fwd60"].mean()), 5),
            "win_rate_fwd60": round(float((sub["fwd60"] > 0).mean()), 4),
            "avg_fwd120": round(float(sub["fwd120"].mean()), 5),
            "win_rate_fwd120": round(float((sub["fwd120"] > 0).mean()), 4)}

    dip = df["ret20"] < -0.08          # sharp 20d drop (>8%)
    oversold = df["pos"] < 0.25         # bottom quartile of the 52w range
    young = df["rl"] < 60               # fresh regime (recent break)
    mature = df["rl"] >= med_age
    cohorts = {
        "baseline_all": _cohort(df["fwd60"].notna(), "all"),
        "sharp_dip_20d(<-8%)": _cohort(dip, "dip"),
        "dip_and_oversold": _cohort(dip & oversold, "dip+oversold"),
        "dip_and_young_regime": _cohort(dip & young, "dip+young"),
        "dip_oversold_young(buy_low_candidate)": _cohort(dip & oversold & young, "full"),
        "dip_in_mature_regime(contrast)": _cohort(dip & mature, "dip+mature"),
    }
    out["buy_low_analysis"] = {k: v for k, v in cohorts.items() if v}
    return out


def _side(a: str) -> str:
    a = (a or "").upper()
    if a in ("BUY", "STRONG_BUY", "OVERWEIGHT"):
        return "L"
    if a in ("SELL", "STRONG_SELL", "UNDERWEIGHT"):
        return "S"
    return "H"


def workflow_overlay(prices: Dict[str, pd.DataFrame], dec: pd.DataFrame,
                     hurst_win: int = 252) -> dict:
    """Attach PIT signals to the crew's real decisions; test flip clustering."""
    # signal as-of each (ticker, run_date)
    sig_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]] = {}
    rows = []
    for _, r in dec.iterrows():
        t = r["ticker"]
        if t not in prices:
            continue
        d = prices[t]
        if t not in sig_cache:
            close = d["Close"].to_numpy(float)
            logret = np.diff(np.log(close), prepend=np.log(close[0]))
            cp, rl = bocpd(logret)
            sig_cache[t] = (close, logret, (cp, rl), d.index)
        close, logret, (cp, rl), idx = sig_cache[t]
        rd = pd.Timestamp(r["run_date"])
        # last bar strictly on/before the run date (PIT)
        pos = idx.searchsorted(rd, side="right") - 1
        if pos < hurst_win + 5 or pos >= len(close):
            continue
        h = dfa_alpha(logret[pos - hurst_win:pos + 1])
        rows.append({"ticker": t, "run_date": r["run_date"], "side": _side(r["action"]),
                     "conviction": r["conviction"], "h": h,
                     "cp": float(cp[pos]) if np.isfinite(cp[pos]) else np.nan,
                     "rl": float(rl[pos]) if np.isfinite(rl[pos]) else np.nan})
    sig = pd.DataFrame(rows).sort_values(["ticker", "run_date"])
    out = {"n_decisions_matched": int(len(sig))}

    # consecutive-day flips: does side change L<->S between adjacent dates?
    pair = []
    for t, sub in sig.groupby("ticker"):
        sub = sub.reset_index(drop=True)
        for i in range(1, len(sub)):
            a, b = sub.loc[i - 1], sub.loc[i]
            if a["side"] in ("L", "S") and b["side"] in ("L", "S"):
                pair.append({"flip": a["side"] != b["side"],
                             "h": b["h"], "cp": b["cp"], "rl": b["rl"],
                             "conviction": b["conviction"]})
    P = pd.DataFrame(pair)
    if len(P):
        P["hb"] = P["h"].apply(hurst_bucket)
        flips_by_h = {}
        for bkt, s in P.groupby("hb"):
            flips_by_h[bkt] = {"n_pairs": int(len(s)),
                               "flip_rate": round(float(s["flip"].mean()), 4)}
        out["flip_rate_by_hurst_regime"] = flips_by_h
        out["overall_pair_flip_rate"] = round(float(P["flip"].mean()), 4)
        # instability (BOCPD run-length) at flips vs non-flips
        fl, nf = P[P.flip], P[~P.flip]
        out["mean_hurst_at_flip_vs_stable"] = {
            "flip": round(float(fl["h"].mean()), 4),
            "no_flip": round(float(nf["h"].mean()), 4)}
        out["mean_regime_age_days_at_flip_vs_stable"] = {
            "flip": round(float(fl["rl"].mean()), 2),
            "no_flip": round(float(nf["rl"].mean()), 2)}
        out["mean_changepoint_prob_at_flip_vs_stable"] = {
            "flip": round(float(fl["cp"].mean()), 5),
            "no_flip": round(float(nf["cp"].mean()), 5)}
    # conviction vs hurst regime (are convictions honest about regime?)
    sig["hb"] = sig["h"].apply(hurst_bucket)
    conv = {}
    for bkt, s in sig.groupby("hb"):
        s = s.dropna(subset=["conviction"])
        if len(s):
            conv[bkt] = {"n": int(len(s)), "avg_conviction": round(float(s["conviction"].mean()), 4)}
    out["conviction_by_hurst_regime"] = conv
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use all 201 names (default: sample)")
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--sample", type=int, default=60)
    args = ap.parse_args()

    uni = universe_from_db()
    print(f"universe in DB: {len(uni)} tickers")
    if not args.full and len(uni) > args.sample:
        # stratified-ish sample: interleave so we get IN (.NS) and US names
        ns = [t for t in uni if t.endswith(".NS")]
        us = [t for t in uni if not t.endswith(".NS")]
        k = args.sample // 2
        uni = ns[:k] + us[:k]
    print(f"using {len(uni)} tickers "
          f"({sum(t.endswith('.NS') for t in uni)} IN / {sum(not t.endswith('.NS') for t in uni)} US)")

    t0 = time.time()
    prices = load_prices(uni, args.years)
    print(f"loaded prices for {len(prices)} tickers in {time.time()-t0:.1f}s")

    print("running efficacy backtest…", flush=True)
    eff = efficacy_backtest(prices)
    print("running workflow overlay…", flush=True)
    dec = decisions_from_db()
    ovl = workflow_overlay(prices, dec)

    result = {"efficacy_backtest": eff, "workflow_overlay": ovl,
              "config": {"tickers": len(prices), "years": args.years}}
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print("\n" + "=" * 70)
    print(json.dumps(result, indent=2, default=str))
    print("=" * 70)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
