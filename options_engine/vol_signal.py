"""
Week 2 — IV Rank + Vol Risk Premium Signal

Two inputs  →  one go/no-go signal for short-premium entry:
  1. IV Rank   : where current ATM IV sits in its 52-week range  (0-100)
  2. VRP       : IV - Realized Vol  (positive = market overpaying for options)

Realized vol uses the Yang-Zhang estimator — handles overnight gaps and
intraday range, ~20% more efficient than close-to-close.

Usage:
    sig = VolSignal("SPY")
    report = sig.evaluate()
    # {'iv_rank': 58.3, 'vrp': 3.1, 'signal': 'SELL', 'confidence': 'HIGH'}
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Thresholds (tune per ticker / strategy) ───────────────────────────────
IV_RANK_MIN = 40    # only sell premium when IV is elevated
VRP_MIN     = 2.0   # IV must exceed RV by at least 2% (annualised)


class VolSignal:
    """
    Computes IV rank + vol risk premium and returns a sell-premium signal.
    Replaces the HV-proxy approach in processors/iv_enricher.py with a
    proper Yang-Zhang RV estimator + clean IV rank calculation.
    """

    def __init__(self, ticker: str, lookback_days: int = 252):
        self.ticker        = ticker.upper()
        self.lookback_days = lookback_days
        self._yf           = yf.Ticker(self.ticker)

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Main entry point — returns the full signal report.
        Feed this into the trade constructor (Week 3) as the go/no-go gate.
        """
        hist  = self._fetch_history()
        rv    = self.yang_zhang_rv(hist, window=21)   # 21-day RV, annualised
        iv_30 = self._atm_iv_30d()
        rank  = self._iv_rank(hist, iv_30)

        vrp       = iv_30 - rv if (iv_30 and rv) else None
        signal, confidence = self._classify(rank, vrp)

        return {
            "ticker":     self.ticker,
            "spot":       round(float(hist["Close"].iloc[-1]), 2),
            "atm_iv_30d": round(iv_30, 2)  if iv_30 else None,
            "rv_21d_yz":  round(rv, 2)     if rv    else None,
            "vrp":        round(vrp, 2)    if vrp   else None,
            "iv_rank":    round(rank, 1)   if rank  else None,
            "signal":     signal,
            "confidence": confidence,
        }

    # ── Yang-Zhang Realized Volatility ───────────────────────────────────

    @staticmethod
    def yang_zhang_rv(hist: pd.DataFrame, window: int = 21) -> float | None:
        """
        Yang-Zhang (2000) estimator — minimum variance, handles gaps + drift.
        Returns annualised vol as a percentage (e.g. 18.5 means 18.5%).

        Formula:
            σ²_YZ = σ²_overnight + k·σ²_open-close + (1-k)·σ²_Rogers-Satchell
            k = 0.34 / (1.34 + (n+1)/(n-1))
        """
        if len(hist) < window + 1:
            return None

        o = hist["Open"].values
        h = hist["High"].values
        l = hist["Low"].values
        c = hist["Close"].values

        # Overnight return: log(Open_t / Close_{t-1})
        ro = np.log(o[1:] / c[:-1])
        # Open-to-close return: log(Close_t / Open_t)
        rc = np.log(c[1:] / o[1:])
        # Rogers-Satchell component
        rs = (np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) +
              np.log(l[1:] / c[1:]) * np.log(l[1:] / o[1:]))

        n  = window
        k  = 0.34 / (1.34 + (n + 1) / (n - 1))

        var_overnight   = pd.Series(ro).rolling(n).var().iloc[-1]
        var_open_close  = pd.Series(rc).rolling(n).var().iloc[-1]
        var_rs          = pd.Series(rs).rolling(n).mean().iloc[-1]

        yz_var = var_overnight + k * var_open_close + (1 - k) * var_rs
        return float(np.sqrt(yz_var * 252) * 100)  # annualised %

    # ── IV Rank ───────────────────────────────────────────────────────────

    def _iv_rank(self, hist: pd.DataFrame, current_iv: float | None) -> float | None:
        """
        IV rank = (current_IV - 52w_low) / (52w_high - 52w_low) * 100
        Uses daily close-to-close HV as IV proxy for the 52w range
        (in production: replace hist_iv_series with actual IV history).
        """
        if current_iv is None:
            return None
        log_ret  = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        hv_series = log_ret.rolling(21).std().dropna() * np.sqrt(252) * 100
        if hv_series.empty:
            return None
        lo, hi = float(hv_series.min()), float(hv_series.max())
        if hi <= lo:
            return 50.0
        return max(0.0, min(100.0, (current_iv - lo) / (hi - lo) * 100))

    # ── Signal classifier ─────────────────────────────────────────────────

    @staticmethod
    def _classify(iv_rank: float | None, vrp: float | None) -> tuple[str, str]:
        if iv_rank is None or vrp is None:
            return "NEUTRAL", "LOW"

        sell  = iv_rank >= IV_RANK_MIN and vrp >= VRP_MIN
        score = (1 if iv_rank >= IV_RANK_MIN else 0) + \
                (1 if iv_rank >= 60         else 0) + \
                (1 if vrp >= VRP_MIN         else 0) + \
                (1 if vrp >= 4.0             else 0)

        if not sell:
            return "NEUTRAL", "LOW"
        confidence = {4: "HIGH", 3: "HIGH", 2: "MED"}.get(score, "LOW")
        return "SELL", confidence

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fetch_history(self) -> pd.DataFrame:
        hist = self._yf.history(period="1y")
        if hist.empty:
            raise ValueError(f"No history for {self.ticker}")
        return hist

    def _atm_iv_30d(self) -> float | None:
        """ATM IV from the 30-DTE options chain using our own BSM IV solver."""
        from datetime import date
        import numpy as np
        from .pricer import iv_solver

        exps = self._yf.options
        if not exps:
            return None
        today  = date.today()
        expiry = min(exps, key=lambda e: abs((date.fromisoformat(e) - today).days - 30))
        try:
            spot = float(self._yf.history(period="1d")["Close"].iloc[-1])
            T    = (date.fromisoformat(expiry) - today).days / 365
            if T <= 0:
                return None
            calls = self._yf.option_chain(expiry).calls
            atm   = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
            mid   = float((atm["bid"] + atm["ask"]).values[0] / 2)
            iv    = iv_solver(
                np.array([mid]), np.array([spot]),
                atm["strike"].values, np.array([T]),
                np.array([0.053]), np.array(["c"])
            )[0]
            return float(iv * 100) if np.isfinite(iv) else None
        except Exception as e:
            log.debug("_atm_iv_30d: %s", e)
            return None
