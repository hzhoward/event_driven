"""
Data feed loader for Backtrader.

Downloads daily OHLCV from yfinance and returns a bt.feeds.PandasData object
ready for cerebro.adddata().  Handles:
  - Tickers that don't exist yet (newer listings)
  - Tickers with partial history (shorter than the full backtest window)
  - Column name variants across yfinance versions

Usage:
    feeds = load_feeds(WATCHLIST, "2022-01-01", "2024-12-31")
    for ticker, feed in feeds.items():
        cerebro.adddata(feed, name=ticker)
"""
from __future__ import annotations
import logging
from typing import Optional

import backtrader as bt
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def load_feed(
    ticker: str,
    start: str,
    end: str,
    min_bars: int = 60,
) -> Optional[bt.feeds.PandasData]:
    """
    Download OHLCV for one ticker and return a BT data feed, or None on failure.

    Parameters
    ----------
    ticker   : e.g. "NVDA"
    start    : ISO date string "YYYY-MM-DD"
    end      : ISO date string "YYYY-MM-DD"
    min_bars : skip ticker if fewer bars available (avoids warm-up errors)
    """
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        log.warning("[%s] yfinance download failed: %s", ticker, e)
        return None

    if raw is None or raw.empty:
        log.warning("[%s] No data returned", ticker)
        return None

    # yfinance ≥0.2.x may return MultiIndex columns — flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalise column names (lower → BT expects these exact names via PandasData)
    raw.columns = [c.lower() for c in raw.columns]
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(raw.columns)
    if missing:
        log.warning("[%s] Missing columns %s", ticker, missing)
        return None

    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw.index = pd.to_datetime(raw.index)

    if len(raw) < min_bars:
        log.warning("[%s] Only %d bars (need %d) — skipping", ticker, len(raw), min_bars)
        return None

    log.info("[%s] Loaded %d bars from %s to %s",
             ticker, len(raw), raw.index[0].date(), raw.index[-1].date())

    return bt.feeds.PandasData(
        dataname=raw,
        datetime=None,   # index IS the datetime
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )


def load_feeds(
    tickers: list[str],
    start: str,
    end: str,
    min_bars: int = 60,
) -> dict[str, bt.feeds.PandasData]:
    """
    Batch-download feeds for every ticker.
    Returns only successfully loaded feeds (skips failed ones with a warning).
    """
    feeds: dict[str, bt.feeds.PandasData] = {}
    for ticker in tickers:
        feed = load_feed(ticker, start, end, min_bars)
        if feed is not None:
            feeds[ticker] = feed
    log.info("Loaded %d / %d tickers", len(feeds), len(tickers))
    return feeds
