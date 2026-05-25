"""
Monday Briefing Generator

Produces a rich-text terminal report + plain-text file.
Format mirrors a real PM morning note:
  - Macro Calendar (FOMC, CPI, NFP)
  - Top Earnings (sorted by impact / IV rank)
  - FDA Binary Events
  - Options Strategy Watchlist
"""
from __future__ import annotations
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config import BRIEFING_OUTPUT_DIR


STRATEGY_COLOR = {
    "Long Straddle":         "bright_green",
    "Long Strangle":         "green",
    "Short Straddle":        "bright_red",
    "Iron Condor":           "yellow",
    "Sell Put":              "red",
    "Bull Put Spread":       "orange1",
    "Long Call":             "bright_cyan",
    "Long Put":              "cyan",
    "Risk Reversal":         "magenta",
    "Close":                 "dim white",
}

IMPACT_COLOR = {10: "bright_red", 9: "red", 8: "orange1",
                7: "yellow", 6: "green", 5: "cyan", 4: "dim cyan",
                3: "dim", 2: "dim", 1: "dim"}


class BriefingGenerator:

    def __init__(self, console: Console | None = None):
        self.console = console or Console(width=120)

    def generate(self, events: list[dict], briefing_date: date) -> str:
        """
        Renders the briefing to the terminal and returns the plain-text version.
        """
        end_date = briefing_date + timedelta(days=14)

        # Partition events
        macro   = [e for e in events if e["event_type"] in
                   ("fomc_decision","fomc_minutes","fed_speech","cpi","ppi",
                    "nfp","gdp","retail_sales","ism","jobless_claims","housing")]
        earnings = [e for e in events if e["event_type"] == "earnings"]
        fda      = [e for e in events if e["event_type"] in ("fda_pdufa","fda_adcom")]
        other    = [e for e in events if e not in macro + earnings + fda]

        # Sort by impact desc, then date
        for lst in (macro, earnings, fda, other):
            lst.sort(key=lambda e: (-e.get("impact_score", 0), e["event_date"]))

        # ── Header ────────────────────────────────────────────────────────
        self.console.print()
        self.console.print(Panel(
            Text.assemble(
                ("Monday Event-Driven Briefing\n", "bold white"),
                (f"Coverage: {briefing_date.strftime('%b %d')} – "
                 f"{end_date.strftime('%b %d, %Y')}", "dim"),
            ),
            style="bold blue",
            box=box.DOUBLE,
        ))

        # ── Macro Calendar ────────────────────────────────────────────────
        if macro:
            self._print_macro_table(macro)

        # ── Earnings ──────────────────────────────────────────────────────
        if earnings:
            self._print_earnings_table(earnings)

        # ── FDA Binary Events ─────────────────────────────────────────────
        if fda:
            self._print_fda_table(fda)

        # ── Strategy Watchlist ────────────────────────────────────────────
        watchlist = [e for e in (macro[:5] + earnings[:15] + fda)
                     if e.get("strategy")]
        if watchlist:
            self._print_strategy_watchlist(watchlist)

        # ── Plain-text output ─────────────────────────────────────────────
        plain = self._build_plain_text(
            briefing_date, end_date, macro, earnings, fda, watchlist
        )
        self._save(plain, briefing_date)
        return plain

    # ── Renderers ────────────────────────────────────────────────────────────

    def _print_macro_table(self, events: list[dict]):
        t = Table(
            title="Macro / Central Bank Calendar",
            box=box.SIMPLE_HEAD,
            title_style="bold yellow",
            show_lines=False,
        )
        t.add_column("Date",       style="dim",    width=12)
        t.add_column("Time ET",    style="dim",    width=10)
        t.add_column("Event",      style="white",  width=35)
        t.add_column("Impact",     justify="center", width=8)
        t.add_column("Strategy",   style="green",  width=28)

        for e in events:
            impact = e.get("impact_score", 0)
            strat_text = _strat_str(e.get("strategy"))
            t.add_row(
                _fmt_date(e["event_date"]),
                e.get("time_et") or "TBD",
                e["title"],
                _impact_badge(impact),
                strat_text,
            )
        self.console.print(t)

    def _print_earnings_table(self, events: list[dict]):
        t = Table(
            title=f"Earnings Calendar  ({len(events)} names)",
            box=box.SIMPLE_HEAD,
            title_style="bold cyan",
            show_lines=False,
        )
        t.add_column("Date",       style="dim",    width=12)
        t.add_column("Ticker",     style="bold",   width=8)
        t.add_column("When",       style="dim",    width=5)
        t.add_column("Impact",     justify="center", width=8)
        t.add_column("IV Rank",    justify="right",  width=9)
        t.add_column("Strategy",   style="green",  width=30)

        for e in events[:25]:  # top 25 by impact
            iv_rank = e.get("iv_rank")
            iv_str  = f"{iv_rank:.0f}  [{e.get('iv_pct_bucket','?').upper()}]" if iv_rank else "—"
            t.add_row(
                _fmt_date(e["event_date"]),
                e.get("ticker") or "—",
                e.get("time_et") or "TBD",
                _impact_badge(e.get("impact_score", 0)),
                iv_str,
                _strat_str(e.get("strategy")),
            )
        self.console.print(t)

    def _print_fda_table(self, events: list[dict]):
        t = Table(
            title="FDA Binary Events",
            box=box.SIMPLE_HEAD,
            title_style="bold magenta",
        )
        t.add_column("Date",      style="dim",    width=12)
        t.add_column("Type",      style="dim",    width=10)
        t.add_column("Ticker",    style="bold",   width=8)
        t.add_column("Event",     style="white",  width=40)
        t.add_column("Strategy",  style="green",  width=25)

        for e in events:
            extra = e.get("extra") or {}
            label = extra.get("drug", "") or e["title"]
            t.add_row(
                _fmt_date(e["event_date"]),
                e["event_type"].upper(),
                e.get("ticker") or "—",
                label[:38],
                _strat_str(e.get("strategy")),
            )
        self.console.print(t)

    def _print_strategy_watchlist(self, events: list[dict]):
        t = Table(
            title="Options Strategy Watchlist",
            box=box.ROUNDED,
            title_style="bold white",
        )
        t.add_column("Date",     style="dim",   width=12)
        t.add_column("Name",     style="white", width=30)
        t.add_column("IV Rank",  justify="right", width=9)
        t.add_column("Pre-Event Strategy",  style="bold green", width=28)
        t.add_column("Entry Guidance",      style="dim",        width=38)

        for e in events:
            iv_rank = e.get("iv_rank")
            iv_str  = f"{iv_rank:.0f}" if iv_rank else "—"
            strat = e.get("strategy") or {}
            t.add_row(
                _fmt_date(e["event_date"]),
                e["title"][:28],
                iv_str,
                strat.get("pre", "—"),
                strat.get("entry", strat.get("note", "—"))[:36],
            )
        self.console.print(t)

    # ── Plain text ────────────────────────────────────────────────────────────
    def _build_plain_text(self, start, end, macro, earnings, fda, watchlist) -> str:
        lines = [
            "=" * 70,
            f"EVENT-DRIVEN MONDAY BRIEFING",
            f"Coverage: {start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}",
            "=" * 70, "",
        ]

        def section(title, events, keys):
            lines.append(f"\n{'─'*60}")
            lines.append(f"  {title}")
            lines.append(f"{'─'*60}")
            for e in events:
                line = f"  {e['event_date']}  {e.get('time_et','TBD'):8s}  " \
                       f"[{e.get('impact_score',0):2d}]  {e['title']}"
                if e.get("ticker"):
                    line += f"  ({e['ticker']})"
                lines.append(line)
                strat = e.get("strategy") or {}
                if strat.get("pre"):
                    lines.append(f"    → Strategy: {strat['pre']}")
                if strat.get("note"):
                    lines.append(f"    → Note: {strat['note']}")

        section("MACRO CALENDAR", macro, [])
        section("EARNINGS (top 25)", earnings[:25], [])
        section("FDA BINARY EVENTS", fda, [])

        lines += ["", "=" * 70,
                  "Generated by event_driven Monday briefing system", "=" * 70]
        return "\n".join(lines)

    def _save(self, content: str, briefing_date: date):
        Path(BRIEFING_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(BRIEFING_OUTPUT_DIR) / f"briefing_{briefing_date.isoformat()}.txt"
        path.write_text(content)
        self.console.print(f"\n[dim]Briefing saved → {path}[/dim]\n")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _fmt_date(date_str: str) -> str:
    try:
        from datetime import datetime
        return datetime.fromisoformat(date_str).strftime("%a %b %d")
    except Exception:
        return date_str


def _impact_badge(score: int) -> Text:
    color = IMPACT_COLOR.get(min(score, 10), "dim")
    return Text("★" * min(score, 5), style=color)


def _strat_str(strategy: Any) -> str:
    if not strategy:
        return "—"
    if isinstance(strategy, dict):
        return strategy.get("pre", "—")
    return str(strategy)
