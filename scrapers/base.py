"""
Shared data model and abstract base class for all scrapers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional
import pytz

ET = pytz.timezone("America/New_York")


class EventType(str, Enum):
    EARNINGS          = "earnings"
    FOMC_DECISION     = "fomc_decision"
    FOMC_MINUTES      = "fomc_minutes"
    FED_SPEECH        = "fed_speech"
    CPI               = "cpi"
    PPI               = "ppi"
    NFP               = "nfp"
    GDP               = "gdp"
    RETAIL_SALES      = "retail_sales"
    ISM               = "ism"
    JOBLESS_CLAIMS    = "jobless_claims"
    HOUSING           = "housing"
    FDA_PDUFA         = "fda_pdufa"
    FDA_ADCOM         = "fda_adcom"
    MA_ANNOUNCEMENT   = "ma_announcement"
    INDEX_REBALANCE   = "index_rebalance"
    INVESTOR_DAY      = "investor_day"
    OTHER             = "other"


@dataclass
class BaseEvent:
    event_type:   EventType
    date:         date
    time_et:      Optional[str]       # "08:30 ET", "BMO", "AMC", "TBD"
    title:        str
    ticker:       Optional[str]       # None for macro events
    description:  str
    impact_score: int                 # 1-10
    source:       str
    source_url:   Optional[str] = None
    extra:        dict = field(default_factory=dict)  # scraper-specific payload

    # Populated by processors downstream
    iv_rank:      Optional[float] = None   # 0-100
    iv_pct_bucket: Optional[str]  = None  # "low" / "med" / "high"
    strategy:     Optional[dict]  = None  # from STRATEGY_MATRIX

    def days_until(self) -> int:
        return (self.date - date.today()).days

    def as_dict(self) -> dict:
        return {
            "event_type":   self.event_type.value,
            "date":         self.date.isoformat(),
            "time_et":      self.time_et,
            "title":        self.title,
            "ticker":       self.ticker,
            "description":  self.description,
            "impact_score": self.impact_score,
            "source":       self.source,
            "source_url":   self.source_url,
            "days_until":   self.days_until(),
            "iv_rank":      self.iv_rank,
            "iv_pct_bucket":self.iv_pct_bucket,
            "strategy":     self.strategy,
            **self.extra,
        }


class ScraperBase:
    """All scrapers implement fetch() → list[BaseEvent]."""
    name: str = "base"

    def fetch(self, start: date, end: date) -> list[BaseEvent]:
        raise NotImplementedError

    @staticmethod
    def _get(url: str, headers: dict | None = None, timeout: int = 15):
        import requests
        hdrs = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        if headers:
            hdrs.update(headers)
        resp = requests.get(url, headers=hdrs, timeout=timeout)
        resp.raise_for_status()
        return resp
