"""
Signal helpers shared between the live trader and the backtester.

Functions
---------
yang_zhang_rv(closes, opens, highs, lows, window)
    Yang-Zhang realized volatility estimator (annualised %, handles overnight gaps).

iv_rank(hv_series, current_iv)
    IV rank on a 0-100 scale relative to 1-year rolling HV series.

momentum_direction(closes, mom_window, trend_window, mom_min_pct)
    Returns "CALL", "PUT", or "NEUTRAL".

strike_for_delta(flag, spot, T, r, target_delta, iv)
    BSM-inverse: find the strike that gives target_delta using brentq.

fetch_history(ticker, lookback_days)
    Download daily OHLCV from yfinance (returns numpy arrays or None).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

log = logging.getLogger(__name__)

RISK_FREE = 0.053   # approximate 3-month T-bill rate


# ── Realized vol ──────────────────────────────────────────────────────────────

def yang_zhang_rv(
    closes: np.ndarray,
    opens:  np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    window: int = 21,
) -> Optional[float]:
    """
    Yang-Zhang realized volatility, annualised as a percentage.
    Requires at least window+1 bars.
    """
    if len(closes) < window + 1:
        return None
    ro = np.log(opens[1:] / closes[:-1])
    rc = np.log(closes[1:] / opens[1:])
    rs = (np.log(highs[1:] / closes[1:]) * np.log(highs[1:] / opens[1:]) +
          np.log(lows[1:]  / closes[1:]) * np.log(lows[1:]  / opens[1:]))
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    ro_var = float(np.var(ro[-window:], ddof=1))
    rc_var = float(np.var(rc[-window:], ddof=1))
    rs_mean = float(np.mean(rs[-window:]))
    yz = ro_var + k * rc_var + (1 - k) * rs_mean
    return float(np.sqrt(max(yz, 0) * 252) * 100)


# ── IV rank ───────────────────────────────────────────────────────────────────

def iv_rank(hv_series: list[float], current_iv: float) -> float:
    """
    IV rank: where does current_iv sit in the range of historical HV values?
    Returns 0–100.  Low rank → cheap vol → buy premium.
    """
    lo = min(hv_series)
    hi = max(hv_series)
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (current_iv - lo) / (hi - lo) * 100))


# ── Momentum direction ────────────────────────────────────────────────────────

def momentum_direction(
    closes:       np.ndarray,
    mom_window:   int   = 20,
    trend_window: int   = 50,
    mom_min_pct:  float = 1.5,
) -> str:
    """
    Returns "CALL" if bullish momentum, "PUT" if bearish, "NEUTRAL" otherwise.

    Rules
    -----
    - mom  = (close[-1] / close[-(mom_window+1)] - 1) × 100
    - sma  = mean(close[-trend_window:])
    - CALL  if mom >  mom_min_pct AND close > sma
    - PUT   if mom < -mom_min_pct AND close < sma
    """
    if len(closes) < trend_window + mom_window:
        return "NEUTRAL"
    mom  = (closes[-1] / closes[-(mom_window + 1)] - 1) * 100
    sma  = float(np.mean(closes[-trend_window:]))
    spot = float(closes[-1])
    if mom >  mom_min_pct and spot > sma:
        return "CALL"
    if mom < -mom_min_pct and spot < sma:
        return "PUT"
    return "NEUTRAL"


# ── BSM helpers ───────────────────────────────────────────────────────────────

def _bsm_price(flag: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if flag == "c" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "c":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bsm_delta(flag: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) if flag == "c" else norm.cdf(d1) - 1)


def strike_for_delta(
    flag:         str,
    spot:         float,
    T:            float,
    r:            float,
    target_delta: float,
    iv:           float,
) -> Optional[float]:
    """
    Find the strike K such that BSM delta(K) = target_delta.
    Uses brentq on [spot×0.30, spot×2.0].
    Returns None if no solution found.
    """
    lo = spot * 0.30
    hi = spot * 2.00
    try:
        f_lo = _bsm_delta(flag, spot, lo, T, r, iv) - target_delta
        f_hi = _bsm_delta(flag, spot, hi, T, r, iv) - target_delta
        if f_lo * f_hi > 0:
            return None
        K = brentq(
            lambda K: _bsm_delta(flag, spot, K, T, r, iv) - target_delta,
            lo, hi, xtol=0.01, maxiter=100,
        )
        return float(K)
    except Exception:
        return None


# ── Historical data fetch ─────────────────────────────────────────────────────

def fetch_history(
    ticker:        str,
    lookback_days: int = 400,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Download daily OHLCV from yfinance and return (closes, opens, highs, lows).
    Returns None if fewer than 60 bars are available.
    """
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "get_level_values"):
            if df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if len(df) < 60:
            return None
        closes = df["close"].values.astype(float)
        opens  = df["open"].values.astype(float)
        highs  = df["high"].values.astype(float)
        lows   = df["low"].values.astype(float)
        return closes, opens, highs, lows
    except Exception as exc:
        log.warning("[%s] history fetch failed: %s", ticker, exc)
        return None


# ── Composite signal ──────────────────────────────────────────────────────────

def compute_signal(
    ticker:       str,
    iv_rank_max:  float = 40.0,
    iv_premium:   float = 1.15,
    mom_window:   int   = 20,
    trend_window: int   = 50,
    mom_min_pct:  float = 1.5,
) -> Optional[dict]:
    """
    Full signal computation for a single ticker.

    Returns dict with keys:
        spot, rv21, atm_iv, ivr, direction, structure
    or None if any prerequisite fails (insufficient data, neutral momentum, IV too high).
    """
    hist = fetch_history(ticker, lookback_days=400)
    if hist is None:
        log.info("[%s] insufficient history", ticker)
        return None

    closes, opens, highs, lows = hist
    spot = float(closes[-1])

    rv21 = yang_zhang_rv(closes, opens, highs, lows, window=21)
    if rv21 is None:
        log.info("[%s] cannot compute RV", ticker)
        return None

    # HV series for IV rank (rolling 21-day std)
    log_ret  = np.log(closes[1:] / closes[:-1])
    hv_series = [
        float(np.std(log_ret[i - 21:i], ddof=1) * np.sqrt(252) * 100)
        for i in range(21, len(log_ret))
    ]
    if len(hv_series) < 10:
        return None

    # IV proxy = YZ-RV × iv_premium (captures typical equity vol risk premium)
    atm_iv = rv21 * iv_premium
    ivr    = iv_rank(hv_series, atm_iv)

    log.info("[%s] spot=%.2f  RV=%.1f%%  IV≈%.1f%%  IVR=%.0f",
             ticker, spot, rv21, atm_iv, ivr)

    if ivr > iv_rank_max:
        log.info("[%s] IV rank %.0f > %.0f — skip", ticker, ivr, iv_rank_max)
        return None

    direction = momentum_direction(closes, mom_window, trend_window, mom_min_pct)
    if direction == "NEUTRAL":
        log.info("[%s] momentum neutral — skip", ticker)
        return None

    structure = "long_call" if (direction == "CALL" and ivr < 20) else \
                "bull_call_spread" if direction == "CALL" else \
                "long_put" if ivr < 20 else "bear_put_spread"

    log.info("[%s] ✓  IVR=%.0f  dir=%s  structure=%s", ticker, ivr, direction, structure)
    return {
        "spot":      spot,
        "rv21":      rv21,
        "atm_iv":    atm_iv,
        "ivr":       ivr,
        "direction": direction,
        "structure": structure,
    }
