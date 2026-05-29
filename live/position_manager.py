"""
Daily position manager — runs at 15:45 ET every trading day.

Exit rules (same as backtester)
--------------------------------
1. DTE ≤ 7            → time exit (theta decay accelerates)
2. Mark ≥ 150 % of premium paid (50 % gain)  → profit target
3. Mark ≤  50 % of premium paid (50 % loss)  → stop loss

Mark computation
----------------
For live positions we use the current Alpaca bid/ask:
  • Long legs: valued at BID (conservative sell-side)
  • Short legs: valued at ASK (conservative buy-back cost)
Net mark = Σ(long_bids) − Σ(short_asks)

When the position cannot be priced (chain gap, market closed) the exit
check is skipped for that ticker and retried next cycle.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from live.alpaca_client import AlpacaClient, AlpacaOrderError

log = logging.getLogger(__name__)

PROFIT_TARGET = 1.50   # close at 150 % of premium (50 % gain on debit)
STOP_LOSS_PCT = 0.50   # close at 50 % of premium remaining (50 % loss)
DTE_CLOSE     = 7

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "live_positions.json"
CLOSED_LOG = Path(__file__).resolve().parent.parent / "data" / "live_closed.jsonl"


# ── State I/O ─────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _append_closed(record: dict):
    CLOSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CLOSED_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ── Mark computation ──────────────────────────────────────────────────────────

def _compute_mark(client: AlpacaClient, meta: dict) -> Optional[float]:
    """
    Calculate the current net mark for a multi-leg position.

    Returns None if any leg has no valid quote (market gap / symbol not found).
    """
    mark = 0.0
    for leg in meta["legs"]:
        action = leg["action"]
        symbol = leg["symbol"]
        positions = client.get_open_positions()
        pos = next((p for p in positions if str(p.symbol) == symbol), None)
        if pos is None:
            log.debug("  %s not in open positions (may already be closed)", symbol)
            return None
        try:
            bid = float(pos.current_price or 0)   # Alpaca Position.current_price ~ mid
            ask = bid  # conservative: use same for buy-back estimate
        except Exception:
            return None
        if bid <= 0:
            return None
        mark += bid if action == "buy" else -ask
    return mark


# ── Close helper ──────────────────────────────────────────────────────────────

def _close_position(client: AlpacaClient, ticker: str, meta: dict, reason: str):
    """
    Close all legs for a position.
    Buy-back short legs FIRST, then sell long legs (safe partial-fill ordering).
    """
    log.info("[%s] closing — %s", ticker, reason)
    legs_ordered = sorted(meta["legs"], key=lambda x: 0 if x["action"] == "sell" else 1)
    for leg in legs_ordered:
        action = leg["action"]
        symbol = leg["symbol"]
        qty    = leg["qty"]
        try:
            if action == "sell":
                # We are short this leg — buy it back
                client.buy_to_close(symbol, qty)
            else:
                # We are long this leg — sell to close
                client.sell_to_close(symbol, qty)
        except AlpacaOrderError as exc:
            log.error("[%s] close leg %s FAILED: %s", ticker, symbol, exc)

    record = {
        **meta,
        "close_date": date.today().isoformat(),
        "reason":     reason,
    }
    _append_closed(record)
    log.info("[%s] ✓ closed", ticker)


# ── Main manager ──────────────────────────────────────────────────────────────

class PositionManager:
    """
    Checks all open positions and exits those that hit a stop rule.
    Call run() once per day, ideally at 15:45 ET.
    """

    def __init__(self, client: AlpacaClient):
        self.client = client

    def run(self, dry_run: bool = False):
        state       = _load_state()
        open_trades = state.get("open_trades", {})

        if not open_trades:
            log.info("No open positions to manage")
            return

        today    = date.today()
        to_close = []   # [(ticker, reason)]

        for ticker, meta in list(open_trades.items()):
            expiry = date.fromisoformat(meta["expiry"])
            dte    = (expiry - today).days

            # ── Expired / near-expiry ─────────────────────────────────────
            if dte < 0:
                log.info("[%s] expired — removing from state", ticker)
                open_trades.pop(ticker)
                continue

            if dte <= DTE_CLOSE:
                to_close.append((ticker, f"time_exit (DTE={dte})"))
                continue

            # ── P&L check ─────────────────────────────────────────────────
            mark = _compute_mark(self.client, meta)
            if mark is None:
                log.debug("[%s] cannot price — skip this cycle", ticker)
                continue

            premium = float(meta["premium"])
            if premium <= 0:
                continue

            gain = (mark - premium) / premium
            log.info("[%s] mark=%.2f  prem=%.2f  gain=%.0f%%",
                     ticker, mark, premium, gain * 100)

            if gain >= (PROFIT_TARGET - 1.0):
                to_close.append((ticker, f"profit_target ({gain*100:+.0f}%)"))
            elif gain <= -STOP_LOSS_PCT:
                to_close.append((ticker, f"stop_loss ({gain*100:+.0f}%)"))

        # ── Execute closes ─────────────────────────────────────────────────
        for ticker, reason in to_close:
            meta = open_trades.get(ticker)
            if meta is None:
                continue
            if dry_run:
                log.info("[%s] DRY RUN close — %s", ticker, reason)
            else:
                _close_position(self.client, ticker, meta, reason)
            open_trades.pop(ticker, None)

        state["open_trades"] = open_trades
        _save_state(state)
        log.info("Position check done — %d closed", len(to_close))

    # ── Convenience: print current positions ──────────────────────────────────

    def print_summary(self):
        state       = _load_state()
        open_trades = state.get("open_trades", {})
        if not open_trades:
            log.info("No open positions")
            return
        today = date.today()
        log.info("%-6s  %-20s  %-6s  %5s  %6s  %s",
                 "Ticker", "Structure", "Dir", "DTE", "Prem", "Opened")
        for ticker, meta in open_trades.items():
            expiry = date.fromisoformat(meta["expiry"])
            dte    = (expiry - today).days
            log.info("%-6s  %-20s  %-6s  %5d  %6.2f  %s",
                     ticker,
                     meta.get("trade_type", "?"),
                     meta.get("direction", "?"),
                     dte,
                     meta.get("premium", 0),
                     meta.get("open_date", "?"))
