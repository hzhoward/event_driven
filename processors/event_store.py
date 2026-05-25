"""
SQLite-backed event store — persists scraped events and supports
idempotent upserts so re-runs on the same Monday don't duplicate rows.
"""
from __future__ import annotations
import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

from scrapers.base import BaseEvent, EventType
from config import DB_PATH

log = logging.getLogger(__name__)


CREATE_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type    TEXT    NOT NULL,
    event_date    TEXT    NOT NULL,
    time_et       TEXT,
    title         TEXT    NOT NULL,
    ticker        TEXT,
    description   TEXT,
    impact_score  INTEGER,
    iv_rank       REAL,
    iv_pct_bucket TEXT,
    strategy_json TEXT,
    source        TEXT,
    source_url    TEXT,
    extra_json    TEXT,
    scraped_at    TEXT    NOT NULL,
    UNIQUE(event_type, event_date, ticker)   -- idempotent upsert key
);

CREATE TABLE IF NOT EXISTS briefings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    briefing_date TEXT    NOT NULL UNIQUE,
    content       TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
"""


class EventStore:

    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(CREATE_DDL)
        self.conn.commit()

    def upsert_events(self, events: list[BaseEvent]) -> int:
        now = datetime.utcnow().isoformat()
        inserted = 0
        for ev in events:
            try:
                self.conn.execute(
                    """
                    INSERT INTO events
                        (event_type, event_date, time_et, title, ticker, description,
                         impact_score, iv_rank, iv_pct_bucket, strategy_json,
                         source, source_url, extra_json, scraped_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_type, event_date, ticker) DO UPDATE SET
                        time_et       = excluded.time_et,
                        description   = excluded.description,
                        impact_score  = excluded.impact_score,
                        iv_rank       = excluded.iv_rank,
                        iv_pct_bucket = excluded.iv_pct_bucket,
                        strategy_json = excluded.strategy_json,
                        extra_json    = excluded.extra_json,
                        scraped_at    = excluded.scraped_at
                    """,
                    (
                        ev.event_type.value,
                        ev.date.isoformat(),
                        ev.time_et,
                        ev.title,
                        ev.ticker,
                        ev.description,
                        ev.impact_score,
                        ev.iv_rank,
                        ev.iv_pct_bucket,
                        json.dumps(ev.strategy) if ev.strategy else None,
                        ev.source,
                        ev.source_url,
                        json.dumps(ev.extra),
                        now,
                    ),
                )
                inserted += 1
            except Exception as exc:
                log.warning("DB upsert failed for %s: %s", ev.title, exc)
        self.conn.commit()
        log.info("EventStore: upserted %d/%d events", inserted, len(events))
        return inserted

    def get_events_in_window(self, start: date, end: date) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM events
            WHERE event_date >= ? AND event_date <= ?
            ORDER BY impact_score DESC, event_date ASC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def save_briefing(self, briefing_date: date, content: str):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO briefings (briefing_date, content, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(briefing_date) DO UPDATE SET
                content    = excluded.content,
                created_at = excluded.created_at
            """,
            (briefing_date.isoformat(), content, now),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("strategy_json", "extra_json"):
        if d.get(k):
            try:
                d[k.replace("_json", "")] = json.loads(d.pop(k))
            except json.JSONDecodeError:
                d.pop(k, None)
    return d
