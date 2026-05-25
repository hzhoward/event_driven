"""
Strategy Mapper — maps each enriched event to a recommended options strategy.

Decision logic mirrors how a PM would think:
  - IV rank (low/med/high) determines whether to buy or sell premium
  - Event type determines direction / structure
  - Days-to-event drives urgency and timing guidance
"""
from __future__ import annotations
from datetime import date

from scrapers.base import BaseEvent, EventType
from config import STRATEGY_MATRIX


# Generic fallbacks when the exact (event_type, iv_bucket) pair is missing
_GENERIC = {
    "low":  {"pre": "Long Straddle",       "post": "Close", "note": "Low IV → buy premium"},
    "med":  {"pre": "Long Strangle (1σ)",  "post": "Iron Condor", "note": "Balanced approach"},
    "high": {"pre": "Iron Condor / Short Straddle", "post": "Close", "note": "High IV → sell premium"},
}


class StrategyMapper:

    def map(self, events: list[BaseEvent]) -> list[BaseEvent]:
        for ev in events:
            bucket = ev.iv_pct_bucket or "med"
            key = (ev.event_type.value, bucket)
            strategy = STRATEGY_MATRIX.get(key) or _GENERIC.get(bucket, _GENERIC["med"])
            ev.strategy = {**strategy, **_timing_guidance(ev)}
        return events


def _timing_guidance(ev: BaseEvent) -> dict:
    """
    Add entry/exit timing to the strategy note based on days-to-event.
    """
    dte = ev.days_until()
    guidance: dict = {}

    if ev.event_type == EventType.EARNINGS:
        if dte >= 10:
            guidance["entry"] = "Consider entering 7-10 DTE when IV not yet fully bid"
            guidance["exit"]  = "Close/flip within 30 min post-announcement; vol crush kills longs"
        elif dte >= 5:
            guidance["entry"] = "Enter now — within the IV ramp window"
            guidance["exit"]  = "Close day of announcement"
        else:
            guidance["entry"] = "< 5 DTE — high IV; premium selling preferred"
            guidance["exit"]  = "Close same day"

    elif ev.event_type in (EventType.FOMC_DECISION, EventType.CPI, EventType.NFP):
        if dte >= 7:
            guidance["entry"] = "Enter 3-5 DTE when macro vol hasn't fully priced in"
            guidance["exit"]  = "Close by EOD of event"
        else:
            guidance["entry"] = "Enter now; set delta-neutral straddle"
            guidance["exit"]  = "Close within 2h of data release"

    elif ev.event_type in (EventType.FDA_PDUFA, EventType.FDA_ADCOM):
        if dte >= 14:
            guidance["entry"] = "Stagger in over next 1-2 weeks; IV will ramp to event"
            guidance["exit"]  = "Close before/on decision day — never hold through binary unknown"
        else:
            guidance["entry"] = "Enter now; final IV ramp underway"
            guidance["exit"]  = "Close on announcement day"

    return guidance
