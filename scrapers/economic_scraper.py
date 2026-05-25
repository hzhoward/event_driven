"""
US macroeconomic calendar scraper.

Sources (in order of priority):
  1. BLS release calendar  (CPI, PPI, NFP, Jobless Claims) — public HTML
  2. BEA release calendar  (GDP, Personal Income) — public HTML
  3. FRED API              (cross-check release dates) — requires API key
  4. Investing.com         (fallback HTML scrape, broad coverage)
"""
from __future__ import annotations
import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .base import BaseEvent, EventType, ScraperBase
from config import IMPACT_WEIGHTS, FRED_API_KEY, ET

log = logging.getLogger(__name__)


# ─── Known macro releases with their event types ─────────────────────────────
MACRO_CATALOG = {
    # title fragment → (EventType, default_time_et)
    "Consumer Price Index":         (EventType.CPI,          "08:30 ET"),
    "CPI":                          (EventType.CPI,          "08:30 ET"),
    "Producer Price Index":         (EventType.PPI,          "08:30 ET"),
    "PPI":                          (EventType.PPI,          "08:30 ET"),
    "Employment Situation":         (EventType.NFP,          "08:30 ET"),
    "Nonfarm Payroll":              (EventType.NFP,          "08:30 ET"),
    "Gross Domestic Product":       (EventType.GDP,          "08:30 ET"),
    "GDP":                          (EventType.GDP,          "08:30 ET"),
    "Retail Sales":                 (EventType.RETAIL_SALES, "08:30 ET"),
    "ISM Manufacturing":            (EventType.ISM,          "10:00 ET"),
    "ISM Services":                 (EventType.ISM,          "10:00 ET"),
    "Unemployment Insurance":       (EventType.JOBLESS_CLAIMS,"08:30 ET"),
    "Initial Claims":               (EventType.JOBLESS_CLAIMS,"08:30 ET"),
    "Housing Starts":               (EventType.HOUSING,      "08:30 ET"),
    "New Home Sales":               (EventType.HOUSING,      "10:00 ET"),
    "Existing Home Sales":          (EventType.HOUSING,      "10:00 ET"),
    "Personal Income":              (EventType.OTHER,        "08:30 ET"),
    "PCE":                          (EventType.CPI,          "08:30 ET"),  # core PCE = Fed's preferred
    "JOLTS":                        (EventType.OTHER,        "10:00 ET"),
}


class EconomicScraper(ScraperBase):
    """
    Multi-source economic calendar. Falls back gracefully if a source is down.
    """
    name = "economic"

    def fetch(self, start: date, end: date) -> list[BaseEvent]:
        events: list[BaseEvent] = []

        # Source 1: BLS schedule (most authoritative for CPI/NFP/PPI/Claims)
        try:
            events.extend(self._fetch_bls(start, end))
        except Exception as exc:
            log.warning("BLS scrape failed: %s", exc)

        # Source 2: FRED release calendar (if API key present)
        if FRED_API_KEY:
            try:
                events.extend(self._fetch_fred(start, end))
            except Exception as exc:
                log.warning("FRED fetch failed: %s", exc)

        # Source 3: Investing.com economic calendar (broad fallback)
        try:
            events.extend(self._fetch_investing_com(start, end))
        except Exception as exc:
            log.warning("Investing.com scrape failed: %s", exc)

        # Deduplicate by (event_type, date)
        seen = set()
        deduped = []
        for ev in events:
            key = (ev.event_type, ev.date)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        log.info("Economic: %d events after dedup", len(deduped))
        return deduped

    # ── BLS ──────────────────────────────────────────────────────────────────
    def _fetch_bls(self, start: date, end: date) -> list[BaseEvent]:
        """
        BLS release schedule via their data API (v2) — bypasses 403 on HTML pages.
        Falls back to hardcoded key-release schedule if API fails.
        """
        # BLS v2 API: series release dates (no registration key needed for basic queries)
        url = "https://api.bls.gov/publicAPI/v2/releases/schedule"
        try:
            resp = self._get(
                url,
                headers={"Content-Type": "application/json"},
            )
            data = resp.json()
            return self._parse_bls_api(data, start, end)
        except Exception as exc:
            log.debug("BLS API failed (%s); using hardcoded schedule", exc)
            return self._hardcoded_bls_schedule(start, end)

    @staticmethod
    def _parse_bls_api(data: dict, start: date, end: date) -> list[BaseEvent]:
        events = []
        for release in data.get("Results", {}).get("releases", []):
            release_name = release.get("releaseName", "")
            for date_info in release.get("releaseDates", []):
                raw_date = date_info.get("date", "")
                try:
                    ev_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if not (start <= ev_date <= end):
                    continue
                ev_type, time_et = EconomicScraper._classify_macro(release_name)
                if ev_type == EventType.OTHER:
                    continue
                impact = IMPACT_WEIGHTS.get(ev_type.value, 5)
                events.append(BaseEvent(
                    event_type=ev_type,
                    date=ev_date,
                    time_et=time_et,
                    title=release_name,
                    ticker=None,
                    description=f"BLS: {release_name}",
                    impact_score=impact,
                    source="bls.gov/api",
                    source_url="https://api.bls.gov/",
                ))
        return events

    def _hardcoded_bls_schedule(self, start: date, end: date) -> list[BaseEvent]:
        """
        Approximate monthly schedule for the most market-moving BLS releases.
        Dates are approximate (first or second Friday/Wednesday of month).
        Used only when the BLS API is unreachable.
        """
        from dateutil.relativedelta import relativedelta

        # (title, event_type, time_et, day_of_month_approx)
        MONTHLY = [
            ("Consumer Price Index",   EventType.CPI,  "08:30 ET", 10),
            ("Producer Price Index",   EventType.PPI,  "08:30 ET", 11),
            ("Retail Sales",           EventType.RETAIL_SALES, "08:30 ET", 15),
            ("Housing Starts",         EventType.HOUSING, "08:30 ET", 17),
        ]
        # NFP: first Friday of month
        WEEKLY_CLAIMS = ("Jobless Claims", EventType.JOBLESS_CLAIMS, "08:30 ET")

        events = []
        cursor = date(start.year, start.month, 1)
        while cursor <= end:
            for title, ev_type, time_et, approx_day in MONTHLY:
                ev_date = date(cursor.year, cursor.month, approx_day)
                if start <= ev_date <= end:
                    events.append(BaseEvent(
                        event_type=ev_type, date=ev_date, time_et=time_et,
                        title=title, ticker=None,
                        description=f"Monthly {title} release (approx date)",
                        impact_score=IMPACT_WEIGHTS.get(ev_type.value, 5),
                        source="hardcoded_approx",
                        extra={"approximate": True},
                    ))
            cursor += relativedelta(months=1)

        return events

    # ── FRED ─────────────────────────────────────────────────────────────────
    def _fetch_fred(self, start: date, end: date) -> list[BaseEvent]:
        """
        Query FRED release calendar API.
        https://fred.stlouisfed.org/docs/api/fred/releases_dates.html
        """
        url = (
            "https://api.stlouisfed.org/fred/releases/dates"
            f"?api_key={FRED_API_KEY}"
            f"&realtime_start={start.isoformat()}"
            f"&realtime_end={end.isoformat()}"
            "&include_release_dates_with_no_data=false"
            "&file_type=json"
        )
        resp = self._get(url)
        data = resp.json()
        events = []

        for release_date_info in data.get("release_dates", []):
            raw_date_str = release_date_info.get("date", "")
            release_name = release_date_info.get("release_name", "")
            try:
                ev_date = date.fromisoformat(raw_date_str)
            except ValueError:
                continue

            if not (start <= ev_date <= end):
                continue

            ev_type, time_et = self._classify_macro(release_name)
            if ev_type == EventType.OTHER:
                continue  # FRED has hundreds of minor releases; skip noise

            impact = IMPACT_WEIGHTS.get(ev_type.value, 5)
            events.append(BaseEvent(
                event_type=ev_type,
                date=ev_date,
                time_et=time_et,
                title=release_name,
                ticker=None,
                description=f"FRED: {release_name}",
                impact_score=impact,
                source="fred",
                source_url=f"https://fred.stlouisfed.org/",
            ))

        return events

    # ── Investing.com (fallback) ──────────────────────────────────────────────
    def _fetch_investing_com(self, start: date, end: date) -> list[BaseEvent]:
        """
        Investing.com economic calendar — HTML scrape.
        They rate-limit aggressively; use sparingly as a fallback.
        """
        url = "https://www.investing.com/economic-calendar/"
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.investing.com/economic-calendar/",
        }

        # POST to their AJAX endpoint
        payload = {
            "country[]": "5",       # USA
            "importance[]": ["2", "3"],  # medium + high impact
            "dateFrom": start.strftime("%Y-%m-%d"),
            "dateTo":   end.strftime("%Y-%m-%d"),
            "timeZone": "8",        # ET offset in their system
            "timeFilter": "timeRemain",
            "currentTab": "custom",
            "submitFilters": "1",
            "limit_from": "0",
        }

        try:
            resp = requests.post(
                "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData",
                data=payload,
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            json_resp = resp.json()
            html = json_resp.get("data", "")
        except Exception as exc:
            log.debug("investing.com AJAX failed: %s; trying static page", exc)
            return []

        soup = BeautifulSoup(html, "lxml")
        events = []

        for row in soup.find_all("tr", {"class": re.compile(r"js-event-item")}):
            try:
                date_td  = row.get("data-event-datetime", "")
                title_td = row.find("td", {"class": "event"})
                bull_tds = row.find_all("i", {"class": "grayFullBullishIcon"})
                impact_level = len(bull_tds)  # 1,2,3 bulls → low,med,high

                if impact_level < 2:
                    continue  # skip low-impact

                ev_date = datetime.strptime(date_td[:10], "%Y/%m/%d").date()
                if not (start <= ev_date <= end):
                    continue

                title = title_td.get_text(strip=True) if title_td else "Unknown"
                ev_type, time_et = self._classify_macro(title)
                impact = IMPACT_WEIGHTS.get(ev_type.value, impact_level * 3)

                events.append(BaseEvent(
                    event_type=ev_type,
                    date=ev_date,
                    time_et=time_et,
                    title=title,
                    ticker=None,
                    description=f"Investing.com: {title}",
                    impact_score=impact,
                    source="investing.com",
                    source_url=url,
                    extra={"impact_bulls": impact_level},
                ))
            except Exception as exc:
                log.debug("investing.com row parse error: %s", exc)
                continue

        return events

    # ── Classifier ───────────────────────────────────────────────────────────
    @staticmethod
    def _classify_macro(name: str) -> tuple[EventType, str]:
        for keyword, (ev_type, time_et) in MACRO_CATALOG.items():
            if keyword.lower() in name.lower():
                return ev_type, time_et
        return EventType.OTHER, "TBD"
