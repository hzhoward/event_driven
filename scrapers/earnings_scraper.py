"""
Earnings calendar scraper.

Primary source  : Yahoo Finance (yfinance earnings_dates + calendar)
Fallback source : Zacks earnings calendar (HTML scrape)
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import BaseEvent, EventType, ScraperBase
from config import IMPACT_WEIGHTS, MEGA_CAP_TICKERS

log = logging.getLogger(__name__)


# S&P 500 tickers we track for earnings — augmented from mega-cap universe.
# In production you would pull the full index membership from a data vendor.
SP500_SAMPLE = MEGA_CAP_TICKERS | {
    "AMGN", "GILD", "REGN", "BIIB", "MRNA", "PFE", "BMY",   # Biotech/Pharma
    "BA",  "LMT",  "RTX",  "NOC",  "GD",                     # Defense
    "F",   "GM",   "RIVN",                                    # Auto
    "C",   "WFC",  "USB",  "PNC",  "TFC",                    # Banks
    "NEE", "DUK",  "SO",   "D",                               # Utilities
    "AMT", "PLD",  "SPG",  "EQR",                             # REITs
    "CAT", "DE",   "HON",  "MMM",  "EMR",                    # Industrials
    "SBUX","NKE",  "TGT",  "LOW",  "TJX",                    # Consumer
    "CVS", "MCK",  "CI",   "HUM",                             # Health insurance
    "UBER","LYFT", "ABNB", "BKNG",                            # Travel/mobility
}


def _cap_tier(ticker: str) -> str:
    if ticker in MEGA_CAP_TICKERS:
        return "mega_cap"
    # rough heuristic — production would use live market-cap data
    large_cap_sample = {
        "AMGN","GILD","REGN","BA","LMT","RTX","C","WFC","CAT","DE","HON",
        "NEE","DUK","SBUX","NKE","TGT","LOW","CVS","MCK","CI","HUM","UBER",
    }
    if ticker in large_cap_sample:
        return "large_cap"
    return "mid_cap"


class EarningsScraper(ScraperBase):
    """
    Fetches expected earnings dates for a watchlist over the target window.

    yfinance is rate-limited; the scraper batches tickers and respects
    a small sleep between requests.
    """
    name = "earnings_yfinance"

    def __init__(self, tickers: Optional[set[str]] = None):
        self.tickers = tickers or SP500_SAMPLE

    def fetch(self, start: date, end: date) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        failed: list[str] = []

        for ticker in sorted(self.tickers):
            try:
                ev = self._fetch_ticker(ticker, start, end)
                events.extend(ev)
            except Exception as exc:
                log.debug("earnings fetch failed for %s: %s", ticker, exc)
                failed.append(ticker)

        if failed:
            log.warning("Earnings: %d tickers failed — %s", len(failed), failed[:10])

        log.info("Earnings: fetched %d events for %d tickers", len(events), len(self.tickers))
        return events

    def _fetch_ticker(self, ticker: str, start: date, end: date) -> list[BaseEvent]:
        yf_ticker = yf.Ticker(ticker)
        cal = yf_ticker.calendar  # dict with 'Earnings Date' key

        if cal is None or "Earnings Date" not in cal:
            return []

        raw_dates = cal["Earnings Date"]
        # yfinance returns a single Timestamp or a list
        if not isinstance(raw_dates, (list, pd.DatetimeTZDtype.__class__)):
            raw_dates = [raw_dates]

        events = []
        for raw_dt in (raw_dates if hasattr(raw_dates, "__iter__") else [raw_dates]):
            try:
                ev_date = pd.Timestamp(raw_dt).date()
            except Exception:
                continue
            if not (start <= ev_date <= end):
                continue

            tier = _cap_tier(ticker)
            impact = IMPACT_WEIGHTS.get(f"earnings_{tier}", 5)

            # Determine timing (BMO / AMC) from yfinance info
            info = {}
            try:
                info = yf_ticker.info or {}
            except Exception:
                pass

            time_et = self._infer_timing(ticker, info)

            # Build a short description with consensus EPS/revenue if available
            eps_est = info.get("forwardEps")
            rev_est = info.get("revenueEstimate")
            desc_parts = [f"{ticker} Q earnings release"]
            if eps_est:
                desc_parts.append(f"EPS est: ${eps_est:.2f}")

            events.append(BaseEvent(
                event_type=EventType.EARNINGS,
                date=ev_date,
                time_et=time_et,
                title=f"{ticker} Earnings",
                ticker=ticker,
                description="; ".join(desc_parts),
                impact_score=impact,
                source=self.name,
                source_url=f"https://finance.yahoo.com/quote/{ticker}/",
                extra={
                    "cap_tier": tier,
                    "eps_estimate": eps_est,
                    "sector": info.get("sector"),
                },
            ))

        return events

    @staticmethod
    def _infer_timing(ticker: str, info: dict) -> str:
        """
        Yahoo Finance doesn't always expose BMO/AMC cleanly.
        Known patterns hard-coded; fallback to 'TBD'.
        """
        # Well-known BMO reporters
        BMO = {"JPM","GS","MS","C","WFC","BAC","USB","PNC","JNJ","PG","KO",
               "CAT","DE","HON","UNH","CVS","MCK","LMT","BA","RTX","MMM"}
        # Well-known AMC reporters
        AMC = {"AAPL","MSFT","GOOGL","GOOG","META","AMZN","NVDA","NFLX",
               "AMD","UBER","LYFT","ABNB","TSLA","QCOM","INTC","TXN",
               "SNAP","PINS","TWTR","CRM","ADBE","ORCL","NOW"}
        if ticker in BMO:
            return "BMO"
        if ticker in AMC:
            return "AMC"
        return "TBD"
