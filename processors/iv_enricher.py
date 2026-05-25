"""
IV Enricher — fetches current IV and 52-week IV percentile for each event's ticker.

Data source: yfinance options chain (free, but limited granularity).
In production you would use CBOE LiveVol, OptionMetrics, or a broker API
(Tastytrade, Interactive Brokers) for real-time IV rank and IV percentile.
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from scrapers.base import BaseEvent

log = logging.getLogger(__name__)

# Macro event tickers for index-level IV
MACRO_TICKERS = {
    "fomc_decision":  "SPY",
    "fomc_minutes":   "SPY",
    "fed_speech":     "SPY",
    "cpi":            "SPY",
    "ppi":            "SPY",
    "nfp":            "SPY",
    "gdp":            "SPY",
    "retail_sales":   "XRT",
    "ism":            "SPY",
    "jobless_claims": "SPY",
    "housing":        "XHB",
}


class IVEnricher:
    """
    Enriches each BaseEvent with:
      - iv_rank       : float 0-100 (IV percentile over 52 weeks)
      - iv_pct_bucket : "low" / "med" / "high"
      - atm_iv        : current at-the-money implied vol (approximate)
    """

    def enrich(self, events: list[BaseEvent]) -> list[BaseEvent]:
        # Collect all unique tickers we need
        tickers_needed: dict[str, list[BaseEvent]] = {}
        for ev in events:
            ticker = ev.ticker or MACRO_TICKERS.get(ev.event_type.value)
            if not ticker:
                continue
            tickers_needed.setdefault(ticker, []).append(ev)

        # Fetch IV data in bulk
        iv_data: dict[str, dict] = {}
        for ticker in tickers_needed:
            try:
                iv_data[ticker] = self._get_iv_metrics(ticker)
            except Exception as exc:
                log.debug("IV fetch failed for %s: %s", ticker, exc)

        # Annotate events
        for ev in events:
            ticker = ev.ticker or MACRO_TICKERS.get(ev.event_type.value)
            if not ticker or ticker not in iv_data:
                ev.iv_pct_bucket = "med"  # conservative default
                continue
            metrics = iv_data[ticker]
            ev.iv_rank = metrics.get("iv_rank")
            ev.extra["atm_iv"] = metrics.get("atm_iv")
            ev.extra["hv_30d"] = metrics.get("hv_30d")
            ev.iv_pct_bucket = _bucket(ev.iv_rank)

        return events

    # ── Core IV calculation ───────────────────────────────────────────────────
    @staticmethod
    def _get_iv_metrics(ticker: str) -> dict:
        """
        Estimate IV rank using the nearest ATM option's implied vol
        and compare to the 52-week range of the VIX proxy (HV).

        yfinance doesn't expose IV rank directly, so we:
          1. Pull 1Y of daily close prices → compute 30-day HV
          2. Fetch the front-month ATM straddle IV from the options chain
          3. Compute IV rank = (current_iv - 52w_low) / (52w_high - 52w_low) * 100
        """
        t = yf.Ticker(ticker)

        # ── Historical vol proxy ──────────────────────────────────────────
        hist = t.history(period="1y")
        if hist.empty:
            return {"iv_rank": 50, "atm_iv": None, "hv_30d": None}

        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        hv_series = log_ret.rolling(21).std() * np.sqrt(252) * 100  # annualised %
        hv_30d    = float(hv_series.iloc[-1]) if not hv_series.empty else None

        # ── ATM IV from options chain ─────────────────────────────────────
        spot = float(hist["Close"].iloc[-1])
        expirations = t.options
        if not expirations:
            return {"iv_rank": 50, "atm_iv": hv_30d, "hv_30d": hv_30d}

        # Use 30-45 day expiry (front month is fine for weekly events)
        target_exp = _nearest_expiry(expirations, days_target=35)
        chain = t.option_chain(target_exp)

        atm_call_iv = _atm_iv(chain.calls, spot)
        atm_put_iv  = _atm_iv(chain.puts, spot)
        atm_iv = (atm_call_iv + atm_put_iv) / 2 if (atm_call_iv and atm_put_iv) else \
                  (atm_call_iv or atm_put_iv or hv_30d or 25.0)

        # ── IV rank: compare current ATM IV to 52-week HV range ──────────
        # (In production, replace with actual 52-week IV history from data vendor)
        hv_52w_low  = float(hv_series.min())
        hv_52w_high = float(hv_series.max())
        if hv_52w_high > hv_52w_low:
            iv_rank = (atm_iv - hv_52w_low) / (hv_52w_high - hv_52w_low) * 100
            iv_rank = max(0.0, min(100.0, iv_rank))
        else:
            iv_rank = 50.0

        return {
            "iv_rank": round(iv_rank, 1),
            "atm_iv":  round(atm_iv, 1),
            "hv_30d":  round(hv_30d, 1) if hv_30d else None,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────
def _nearest_expiry(expirations: tuple, days_target: int = 35) -> str:
    today = date.today()
    best, best_delta = expirations[0], float("inf")
    for exp in expirations:
        try:
            exp_date = date.fromisoformat(exp)
            delta = abs((exp_date - today).days - days_target)
            if delta < best_delta:
                best, best_delta = exp, delta
        except ValueError:
            continue
    return best


def _atm_iv(chain: pd.DataFrame, spot: float) -> Optional[float]:
    if chain.empty:
        return None
    chain = chain.copy()
    chain["dist"] = (chain["strike"] - spot).abs()
    atm_row = chain.nsmallest(1, "dist")
    if atm_row.empty:
        return None
    iv = atm_row["impliedVolatility"].iloc[0]
    return float(iv * 100) if iv > 0 else None


def _bucket(iv_rank: Optional[float]) -> str:
    if iv_rank is None:
        return "med"
    if iv_rank < 25:
        return "low"
    if iv_rank < 75:
        return "med"
    return "high"
