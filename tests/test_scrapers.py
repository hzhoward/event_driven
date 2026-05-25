"""
Unit tests — run with: pytest tests/ -v
These test data parsing logic without hitting live endpoints.
"""
from datetime import date
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scrapers.base import BaseEvent, EventType
from scrapers.fomc_scraper import FOMCScraper
from scrapers.economic_scraper import EconomicScraper
from scrapers.fda_scraper import _parse_fda_date, _extract_ticker_hint
from processors.strategy_mapper import StrategyMapper
from processors.iv_enricher import _bucket


# ── FOMC hardcoded calendar ───────────────────────────────────────────────
def test_fomc_returns_decision_events():
    scraper = FOMCScraper()
    events = scraper._from_hardcoded(date(2025, 5, 1), date(2025, 12, 31))
    assert len(events) > 0
    decision_events = [e for e in events if e.event_type == EventType.FOMC_DECISION]
    assert len(decision_events) >= 4  # at least 4 meetings in the window


def test_fomc_minutes_dates_present():
    scraper = FOMCScraper()
    events = scraper._from_hardcoded(date(2025, 1, 1), date(2025, 12, 31))
    minutes = [e for e in events if e.event_type == EventType.FOMC_MINUTES]
    assert len(minutes) > 0


def test_fomc_respects_date_window():
    scraper = FOMCScraper()
    events = scraper._from_hardcoded(date(2025, 6, 17), date(2025, 6, 18))
    assert all(date(2025, 6, 17) <= e.date <= date(2025, 6, 18) for e in events)


# ── Economic classifier ───────────────────────────────────────────────────
def test_economic_classifier_cpi():
    ev_type, time_et = EconomicScraper._classify_macro("Consumer Price Index - All Urban")
    assert ev_type == EventType.CPI
    assert "08:30" in time_et


def test_economic_classifier_nfp():
    ev_type, _ = EconomicScraper._classify_macro("Employment Situation Summary")
    assert ev_type == EventType.NFP


def test_economic_classifier_unknown():
    ev_type, _ = EconomicScraper._classify_macro("Widget Shipment Report Q3")
    assert ev_type == EventType.OTHER


# ── FDA helpers ───────────────────────────────────────────────────────────
def test_parse_fda_date_formats():
    assert _parse_fda_date("January 15, 2025") == date(2025, 1, 15)
    assert _parse_fda_date("01/15/2025")        == date(2025, 1, 15)
    assert _parse_fda_date("2025-01-15")         == date(2025, 1, 15)
    assert _parse_fda_date("Jan 15, 2025")       == date(2025, 1, 15)
    assert _parse_fda_date("garbage")            is None


def test_extract_ticker_hint():
    assert _extract_ticker_hint("Moderna (MRNA) BLA review") == "MRNA"
    assert _extract_ticker_hint("Pfizer (PFE)")               == "PFE"
    # Should not extract noise words
    result = _extract_ticker_hint("FDA Advisory Committee")
    assert result not in ("FDA", "THE", "FOR")


# ── Strategy mapper ───────────────────────────────────────────────────────
def _make_event(ev_type, iv_bucket="med", days=10):
    ev = BaseEvent(
        event_type=ev_type,
        date=date.today().__class__.today() + __import__("datetime").timedelta(days=days),
        time_et="08:30 ET",
        title="Test Event",
        ticker="SPY",
        description="",
        impact_score=8,
        source="test",
    )
    ev.iv_pct_bucket = iv_bucket
    return ev


def test_strategy_mapper_earnings_low_iv():
    mapper = StrategyMapper()
    ev = _make_event(EventType.EARNINGS, "low", days=10)
    [mapped] = mapper.map([ev])
    assert mapped.strategy is not None
    assert "Straddle" in mapped.strategy.get("pre", "")


def test_strategy_mapper_fomc_high_iv():
    mapper = StrategyMapper()
    ev = _make_event(EventType.FOMC_DECISION, "high", days=3)
    [mapped] = mapper.map([ev])
    assert "Iron Condor" in mapped.strategy.get("pre", "") or \
           "Short" in mapped.strategy.get("pre", "")


def test_strategy_mapper_fda_low_iv():
    mapper = StrategyMapper()
    ev = _make_event(EventType.FDA_PDUFA, "low", days=12)
    [mapped] = mapper.map([ev])
    assert "Straddle" in mapped.strategy.get("pre", "")


# ── IV bucket ─────────────────────────────────────────────────────────────
def test_iv_bucket():
    assert _bucket(10)  == "low"
    assert _bucket(50)  == "med"
    assert _bucket(90)  == "high"
    assert _bucket(None)== "med"
    assert _bucket(25)  == "med"
    assert _bucket(24.9)== "low"
