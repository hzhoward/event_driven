#!/usr/bin/env python3
"""
Event-Driven Monday Briefing — main entry point.

Usage:
    python main.py                        # run today (or next Monday)
    python main.py --date 2025-06-02      # run for a specific Monday
    python main.py --no-iv                # skip IV enrichment (faster / offline)
    python main.py --tickers AAPL,MSFT    # override earnings watchlist
    python main.py --daemon               # run every Monday at 07:00 ET
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import date, timedelta

import schedule
import time

from scrapers import EarningsScraper, EconomicScraper, FDAScraper, FOMCScraper
from processors import IVEnricher, StrategyMapper, EventStore
from reporters import BriefingGenerator
from config import LOOKAHEAD_DAYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("briefing")


def run_briefing(
    briefing_date: date,
    skip_iv: bool = False,
    tickers: set[str] | None = None,
) -> str:
    start = briefing_date
    end   = briefing_date + timedelta(days=LOOKAHEAD_DAYS)

    log.info("Running briefing for %s → %s", start, end)

    # ── 1. Scrape ─────────────────────────────────────────────────────────
    all_events = []

    log.info("Scraping earnings …")
    earnings_scraper = EarningsScraper(tickers=tickers)
    all_events += earnings_scraper.fetch(start, end)

    log.info("Scraping economic calendar …")
    all_events += EconomicScraper().fetch(start, end)

    log.info("Scraping FOMC calendar …")
    all_events += FOMCScraper().fetch(start, end)

    log.info("Scraping FDA calendar …")
    all_events += FDAScraper().fetch(start, end)

    log.info("Total raw events: %d", len(all_events))

    # ── 2. Enrich with IV ─────────────────────────────────────────────────
    if not skip_iv:
        log.info("Enriching with IV rank (this takes ~60s for a full watchlist) …")
        all_events = IVEnricher().enrich(all_events)
    else:
        log.info("IV enrichment skipped")
        for ev in all_events:
            ev.iv_pct_bucket = "med"

    # ── 3. Map to options strategies ──────────────────────────────────────
    all_events = StrategyMapper().map(all_events)

    # ── 4. Persist ────────────────────────────────────────────────────────
    store = EventStore()
    store.upsert_events(all_events)
    enriched_events = store.get_events_in_window(start, end)
    log.info("Loaded %d events from store", len(enriched_events))

    # ── 5. Generate briefing ──────────────────────────────────────────────
    generator = BriefingGenerator()
    briefing_text = generator.generate(enriched_events, briefing_date)
    store.save_briefing(briefing_date, briefing_text)
    store.close()

    return briefing_text


def _next_monday(today: date) -> date:
    days_ahead = (7 - today.weekday()) % 7
    return today if today.weekday() == 0 else today + timedelta(days=days_ahead)


def main():
    parser = argparse.ArgumentParser(description="Event-Driven Monday Briefing")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Briefing start date YYYY-MM-DD (default: today or next Monday)",
    )
    parser.add_argument(
        "--no-iv", action="store_true",
        help="Skip IV enrichment (faster, no options chain calls)",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated additional tickers for earnings watchlist",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as daemon — generate briefing every Monday at 07:00 ET",
    )
    args = parser.parse_args()

    extra_tickers = set(args.tickers.split(",")) if args.tickers else None

    if args.date:
        briefing_date = date.fromisoformat(args.date)
    else:
        briefing_date = _next_monday(date.today())

    if args.daemon:
        log.info("Daemon mode — scheduling Monday 07:00 ET briefings")

        def _job():
            run_briefing(date.today(), skip_iv=args.no_iv, tickers=extra_tickers)

        schedule.every().monday.at("07:00").do(_job)
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_briefing(briefing_date, skip_iv=args.no_iv, tickers=extra_tickers)


if __name__ == "__main__":
    main()
