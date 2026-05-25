"""
Federal Reserve / FOMC event scraper.

Sources:
  1. Federal Reserve meeting schedule page (authoritative)
  2. Hard-coded 2024-2026 calendar as fallback (updated annually)
"""
from __future__ import annotations
import logging
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from .base import BaseEvent, EventType, ScraperBase
from config import IMPACT_WEIGHTS

log = logging.getLogger(__name__)


# ─── Hard-coded FOMC calendar (authoritative backup) ──────────────────────
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Last updated: 2025.  Update each January.
FOMC_CALENDAR_HARDCODED = [
    # 2025
    {"date": "2025-01-28/29", "decision": True,  "minutes": "2025-02-19"},
    {"date": "2025-03-18/19", "decision": True,  "minutes": "2025-04-09"},
    {"date": "2025-05-06/07", "decision": True,  "minutes": "2025-05-28"},
    {"date": "2025-06-17/18", "decision": True,  "minutes": "2025-07-09"},
    {"date": "2025-07-29/30", "decision": True,  "minutes": "2025-08-20"},
    {"date": "2025-09-16/17", "decision": True,  "minutes": "2025-10-08"},
    {"date": "2025-10-28/29", "decision": True,  "minutes": "2025-11-19"},
    {"date": "2025-12-09/10", "decision": True,  "minutes": "2026-01-07"},
    # 2026
    {"date": "2026-01-27/28", "decision": True,  "minutes": "2026-02-18"},
    {"date": "2026-03-17/18", "decision": True,  "minutes": "2026-04-08"},
    {"date": "2026-04-28/29", "decision": True,  "minutes": "2026-05-20"},
    {"date": "2026-06-09/10", "decision": True,  "minutes": "2026-07-01"},
    {"date": "2026-07-28/29", "decision": True,  "minutes": "2026-08-19"},
    {"date": "2026-09-15/16", "decision": True,  "minutes": "2026-10-07"},
    {"date": "2026-10-27/28", "decision": True,  "minutes": "2026-11-18"},
    {"date": "2026-12-15/16", "decision": True,  "minutes": "2027-01-06"},
]


class FOMCScraper(ScraperBase):
    """
    Fetches FOMC meeting (decision day) and minutes release dates.
    Decision is announced at ~14:00 ET on the second day of each 2-day meeting.
    """
    name = "fomc"
    FED_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def fetch(self, start: date, end: date) -> list[BaseEvent]:
        events: list[BaseEvent] = []

        try:
            events = self._fetch_live(start, end)
            if events:
                log.info("FOMC: fetched %d events from Fed website", len(events))
                return events
        except Exception as exc:
            log.warning("FOMC live fetch failed (%s); using hardcoded calendar", exc)

        return self._from_hardcoded(start, end)

    # ── Live scrape ──────────────────────────────────────────────────────────
    def _fetch_live(self, start: date, end: date) -> list[BaseEvent]:
        resp = self._get(self.FED_URL)
        soup = BeautifulSoup(resp.text, "lxml")
        events = []

        # The Fed page lists meeting dates in panels by year
        for panel in soup.find_all("div", {"class": "panel"}):
            year_heading = panel.find("h4")
            if not year_heading:
                continue
            year_match = re.search(r"(\d{4})", year_heading.text)
            if not year_match:
                continue
            year = int(year_match.group(1))

            for item in panel.find_all("li"):
                text = item.get_text(" ", strip=True)
                # Parse date ranges like "January 28-29" or "March 18-19*"
                m = re.search(
                    r"(January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\s+(\d+)[-–](\d+)",
                    text,
                )
                if not m:
                    continue
                month_str, day1_str, day2_str = m.group(1), m.group(2), m.group(3)
                try:
                    decision_date = datetime.strptime(
                        f"{month_str} {day2_str} {year}", "%B %d %Y"
                    ).date()
                except ValueError:
                    continue

                # Check if this is a projected (starred) unscheduled meeting
                is_scheduled = "*" not in text

                if not (start <= decision_date <= end):
                    # Also check minutes date embedded in the same li
                    pass
                else:
                    events.append(_make_decision_event(decision_date, is_scheduled))

                # Minutes date sometimes listed inline as "Minutes: Month DD"
                min_m = re.search(r"Minutes[:\s]+(\w+ \d+)", text)
                if min_m:
                    try:
                        mins_date = datetime.strptime(
                            f"{min_m.group(1)} {year}", "%B %d %Y"
                        ).date()
                        if start <= mins_date <= end:
                            events.append(_make_minutes_event(mins_date, decision_date))
                    except ValueError:
                        pass

        return events

    # ── Hardcoded fallback ────────────────────────────────────────────────────
    def _from_hardcoded(self, start: date, end: date) -> list[BaseEvent]:
        events = []
        for entry in FOMC_CALENDAR_HARDCODED:
            # Decision date = second day of meeting range
            end_day_str = entry["date"].split("/")[1]
            base_str = entry["date"].split("/")[0]  # "YYYY-MM-DD"
            year_month = base_str[:8]               # "YYYY-MM-"
            decision_str = year_month + end_day_str.zfill(2)
            try:
                decision_date = date.fromisoformat(decision_str)
            except ValueError:
                continue
            if start <= decision_date <= end:
                events.append(_make_decision_event(decision_date, scheduled=True))

            mins_str = entry.get("minutes")
            if mins_str:
                try:
                    mins_date = date.fromisoformat(mins_str)
                    if start <= mins_date <= end:
                        events.append(_make_minutes_event(mins_date, decision_date))
                except ValueError:
                    pass
        return events


def _make_decision_event(decision_date: date, scheduled: bool) -> BaseEvent:
    return BaseEvent(
        event_type=EventType.FOMC_DECISION,
        date=decision_date,
        time_et="14:00 ET",
        title="FOMC Rate Decision",
        ticker=None,
        description=(
            "Federal Reserve interest rate decision. "
            "Market-moving for SPY/QQQ/TLT/USD. "
            "Press conference follows at 14:30 ET."
        ),
        impact_score=IMPACT_WEIGHTS["fomc_decision"],
        source="federalreserve.gov",
        source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        extra={"scheduled": scheduled, "press_conference": True},
    )


def _make_minutes_event(mins_date: date, meeting_date: date) -> BaseEvent:
    return BaseEvent(
        event_type=EventType.FOMC_MINUTES,
        date=mins_date,
        time_et="14:00 ET",
        title="FOMC Minutes Release",
        ticker=None,
        description=(
            f"Minutes from the {meeting_date.strftime('%b %d')} FOMC meeting. "
            "Can reveal hawkish/dovish nuances not in the statement."
        ),
        impact_score=IMPACT_WEIGHTS["fomc_minutes"],
        source="federalreserve.gov",
        source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        extra={"meeting_date": meeting_date.isoformat()},
    )
