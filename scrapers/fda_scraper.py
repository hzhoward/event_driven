"""
FDA event scraper — PDUFA dates and Advisory Committee meetings.

Sources:
  1. FDA PDUFA action dates (scraped from FDA.gov) — binary drug approval events
  2. FDA Advisory Committee calendar — foreshadows PDUFA outcomes

These are the highest-impact binary events in healthcare/biotech options.
"""
from __future__ import annotations
import logging
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from .base import BaseEvent, EventType, ScraperBase
from config import IMPACT_WEIGHTS

log = logging.getLogger(__name__)


class FDAScraper(ScraperBase):
    """
    Scrapes FDA PDUFA calendar and AdCom schedule from FDA.gov.
    Falls back to BioPharma Catalyst if FDA is unreachable.
    """
    name = "fda"

    FDA_PDUFA_URL   = "https://www.fda.gov/patients/drug-development-process/step-4-fda-drug-review"
    FDA_ADCOM_URL   = "https://www.fda.gov/advisory-committees/advisory-committee-calendar"
    BIOPHARM_URL    = "https://biopharma-catalyst.com/fda-pdufa-calendar/"

    def fetch(self, start: date, end: date) -> list[BaseEvent]:
        events: list[BaseEvent] = []

        try:
            events.extend(self._fetch_adcom(start, end))
        except Exception as exc:
            log.warning("FDA AdCom scrape failed: %s", exc)

        try:
            events.extend(self._fetch_biopharma_catalyst(start, end))
        except Exception as exc:
            log.warning("BioPharma Catalyst scrape failed: %s", exc)

        # Deduplicate by (ticker, date)
        seen = set()
        deduped = []
        for ev in events:
            key = (ev.ticker, ev.date, ev.event_type)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        log.info("FDA: %d events", len(deduped))
        return deduped

    # ── FDA Advisory Committee Calendar ──────────────────────────────────────
    def _fetch_adcom(self, start: date, end: date) -> list[BaseEvent]:
        resp = self._get(self.FDA_ADCOM_URL)
        soup = BeautifulSoup(resp.text, "lxml")
        events = []

        # FDA AdCom page lists meetings in tables with date + meeting title
        for row in soup.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) < 2:
                continue
            date_text  = cols[0].get_text(strip=True)
            title_text = cols[1].get_text(strip=True) if len(cols) > 1 else ""

            ev_date = _parse_fda_date(date_text)
            if ev_date is None or not (start <= ev_date <= end):
                continue

            # Try to extract ticker from drug name / company
            ticker = _extract_ticker_hint(title_text)
            events.append(BaseEvent(
                event_type=EventType.FDA_ADCOM,
                date=ev_date,
                time_et="09:00 ET",
                title=f"FDA AdCom: {title_text[:80]}",
                ticker=ticker,
                description=title_text,
                impact_score=IMPACT_WEIGHTS["fda_adcom"],
                source="fda.gov/adcom",
                source_url=self.FDA_ADCOM_URL,
            ))

        return events

    # ── BioPharma Catalyst PDUFA page ────────────────────────────────────────
    def _fetch_biopharma_catalyst(self, start: date, end: date) -> list[BaseEvent]:
        """
        BioPharma Catalyst aggregates PDUFA dates from SEC filings and FDA press releases.
        Table columns: Date | Company | Drug | Indication | Ticker
        """
        resp = self._get(self.BIOPHARM_URL)
        soup = BeautifulSoup(resp.text, "lxml")
        events = []

        table = soup.find("table", {"id": re.compile(r"pdufa", re.I)}) or \
                soup.find("table", {"class": re.compile(r"pdufa|catalyst", re.I)}) or \
                soup.find("table")

        if not table:
            log.debug("BioPharma Catalyst: no table found")
            return events

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        date_col    = _col_idx(headers, ["date", "pdufa date"])
        company_col = _col_idx(headers, ["company"])
        drug_col    = _col_idx(headers, ["drug", "candidate"])
        ticker_col  = _col_idx(headers, ["ticker", "symbol"])
        indication_col = _col_idx(headers, ["indication", "disease"])

        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if not cols:
                continue
            try:
                raw_date  = cols[date_col].get_text(strip=True)    if date_col    is not None else ""
                company   = cols[company_col].get_text(strip=True) if company_col is not None else ""
                drug      = cols[drug_col].get_text(strip=True)    if drug_col    is not None else ""
                ticker    = cols[ticker_col].get_text(strip=True)  if ticker_col  is not None else None
                indication= cols[indication_col].get_text(strip=True) if indication_col is not None else ""

                ev_date = _parse_fda_date(raw_date)
                if ev_date is None or not (start <= ev_date <= end):
                    continue

                if not ticker:
                    ticker = _extract_ticker_hint(company)

                desc = f"{company} — {drug}"
                if indication:
                    desc += f" ({indication})"

                events.append(BaseEvent(
                    event_type=EventType.FDA_PDUFA,
                    date=ev_date,
                    time_et="TBD",
                    title=f"FDA PDUFA: {company} {drug}",
                    ticker=ticker.upper() if ticker else None,
                    description=desc,
                    impact_score=IMPACT_WEIGHTS["fda_pdufa"],
                    source="biopharma-catalyst.com",
                    source_url=self.BIOPHARM_URL,
                    extra={
                        "company":    company,
                        "drug":       drug,
                        "indication": indication,
                    },
                ))
            except (IndexError, AttributeError) as exc:
                log.debug("PDUFA row parse error: %s", exc)

        return events


# ── Helpers ──────────────────────────────────────────────────────────────────
def _parse_fda_date(text: str) -> date | None:
    """Try several date formats used by FDA and related sites."""
    text = text.strip().rstrip("*†‡")
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d",
                "%B %Y", "%b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    # e.g. "Q2 2025" → skip
    return None


def _extract_ticker_hint(text: str) -> str | None:
    """
    Look for an all-caps 1-5 letter ticker pattern in parentheses or standalone.
    E.g.  "Pfizer (PFE)" → "PFE"
    """
    m = re.search(r"\b([A-Z]{1,5})\b", text)
    if m:
        candidate = m.group(1)
        # Exclude common English words that happen to be uppercase
        noise = {"FDA","NDA","BLA","PDUFA","IND","ADCOM","THE","FOR","AND","OR","IN"}
        if candidate not in noise:
            return candidate
    return None


def _col_idx(headers: list[str], candidates: list[str]) -> int | None:
    for c in candidates:
        for i, h in enumerate(headers):
            if c in h:
                return i
    return None
