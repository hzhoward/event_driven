"""
Monday evaluator — integrates event scraper with vol signal and Alpaca orders.

Flow
----
1.  Run all scrapers for the coming week to find catalysts.
2.  Build a prioritised ticker list:
    a) Tickers with high-impact events in the next 5–14 days (earnings,
       FOMC, CPI/NFP, FDA PDUFA).
    b) Fallback: full WATCHLIST (same as backtester).
3.  For each candidate (until MAX_POSITIONS filled):
    a) compute_signal() → check IV rank + momentum.
    b) Query Alpaca option chain for the best-DTE expiry.
    c) Find long leg strike (nearest delta to 0.40).
    d) For spreads, find short leg strike (SPREAD_WING_PCT above/below).
    e) Size by MAX_PREMIUM_PCT of account value.
    f) Submit long leg first (BTO), then short leg (STO).
    g) Record open trade in state file.
4.  Persist state to JSON so position_manager can monitor exits.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from live.alpaca_client import AlpacaClient, AlpacaOrderError, OptionContract
from live.signal import compute_signal, strike_for_delta, RISK_FREE

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

WATCHLIST = [
    "SPY",  "QQQ",  "IWM",
    "NVDA", "AMD",  "AVGO", "MSFT", "META",
    "PLTR", "ANET", "VRT",  "GEV",  "CRDO",
    "TSM",  "ORCL", "PANW", "NOW",  "ASML",
    "AMAT", "NFLX", "NEE",  "EQIX",
    "DLR",  "AMT",  "TSLA", "WDC",  "INTC",
]

MAX_POSITIONS    = 5
MAX_PREMIUM_PCT  = 0.02    # risk at most 2 % of NAV per position
IV_RANK_MAX      = 40.0
LONG_DELTA       = 0.40    # near-ATM for maximum gamma
SPREAD_WING_PCT  = 0.03    # 3 % of spot for spread wing
DTE_TARGET       = 30      # ideal days-to-expiry
DTE_MIN          = 15
DTE_MAX          = 45

# Event types that make a ticker "catalyst-ready" (eligible for priority entry)
_CATALYST_TYPES = {
    "earnings", "fomc_decision", "cpi", "nfp", "fda_pdufa", "fda_adcom",
}
_CATALYST_HORIZON_DAYS = 14   # look for events within the next N days
_CATALYST_IMPACT_MIN   = 6    # minimum impact score to qualify

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "live_positions.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _event_tickers(lookahead: int = _CATALYST_HORIZON_DAYS) -> list[str]:
    """
    Run all scrapers and return a deduplicated list of tickers that have
    high-impact catalyst events in the next *lookahead* days.
    """
    try:
        from scrapers import EarningsScraper, EconomicScraper, FDAScraper, FOMCScraper
    except ImportError:
        log.warning("Scrapers not importable — using pure WATCHLIST")
        return []

    today = date.today()
    end   = today + timedelta(days=lookahead)

    tickers: list[str] = []
    try:
        for ev in EarningsScraper().fetch(today, end):
            if (ev.ticker and
                ev.impact_score >= _CATALYST_IMPACT_MIN and
                ev.event_type.value in _CATALYST_TYPES):
                tickers.append(ev.ticker)
    except Exception as exc:
        log.warning("EarningsScraper failed: %s", exc)

    try:
        for ev in FDAScraper().fetch(today, end):
            if (ev.ticker and
                ev.impact_score >= _CATALYST_IMPACT_MIN and
                ev.event_type.value in _CATALYST_TYPES):
                tickers.append(ev.ticker)
    except Exception as exc:
        log.warning("FDAScraper failed: %s", exc)

    # Macro events (FOMC/CPI/NFP) apply to index ETFs
    try:
        macro = [e for e in EconomicScraper().fetch(today, end)
                 if e.event_type.value in _CATALYST_TYPES and
                    e.impact_score >= _CATALYST_IMPACT_MIN]
        if macro:
            tickers.extend(["SPY", "QQQ", "IWM"])
    except Exception as exc:
        log.warning("EconomicScraper failed: %s", exc)

    try:
        fomc = [e for e in FOMCScraper().fetch(today, end)
                if e.impact_score >= _CATALYST_IMPACT_MIN]
        if fomc:
            tickers.extend(["SPY", "QQQ", "IWM"])
    except Exception as exc:
        log.warning("FOMCScraper failed: %s", exc)

    # Deduplicate preserving order; only keep tickers also in WATCHLIST
    seen: set[str] = set()
    result: list[str] = []
    for t in tickers:
        if t not in seen and t in WATCHLIST:
            seen.add(t)
            result.append(t)
    log.info("Catalyst tickers: %s", result)
    return result


def _nearest_contract(
    contracts:    list[OptionContract],
    right:        str,
    target_delta: float,
    spot:         float,
    T:            float,
    iv:           float,
) -> Optional[OptionContract]:
    """
    Return the contract whose strike is closest to the target-delta strike.
    Filters by right ("call"/"put") and requires positive ask.
    """
    filtered = [c for c in contracts
                if c.right == right and c.ask_price > 0]
    if not filtered:
        return None

    # Compute target strike via BSM inversion
    flag   = "c" if right == "call" else "p"
    k_star = strike_for_delta(flag, spot, T, RISK_FREE, target_delta, iv / 100.0)
    if k_star is None:
        # Fallback: use spot × (1 ± 0.02) as a rough ATM strike
        k_star = spot * (0.98 if right == "put" else 1.02)

    best = min(filtered, key=lambda c: abs(c.strike - k_star))
    return best


def _choose_expiry_contracts(
    chain: list[OptionContract],
) -> list[OptionContract]:
    """
    From the full chain, keep only contracts for the single expiry
    that is closest to DTE_TARGET days from today.
    """
    today = date.today()
    expiries = sorted({c.expiry for c in chain})
    if not expiries:
        return []
    best_exp = min(expiries, key=lambda e: abs((e - today).days - DTE_TARGET))
    return [c for c in chain if c.expiry == best_exp]


# ── Leg builders ──────────────────────────────────────────────────────────────

def _build_long_call(
    contracts: list[OptionContract],
    spot: float, T: float, iv: float,
) -> Optional[dict]:
    lc = _nearest_contract(contracts, "call", LONG_DELTA, spot, T, iv)
    if lc is None:
        return None
    premium = round(lc.ask_price, 2)
    if premium <= 0:
        return None
    return {
        "legs":    [{"action": "buy", "contract": lc}],
        "premium": premium,
    }


def _build_long_put(
    contracts: list[OptionContract],
    spot: float, T: float, iv: float,
) -> Optional[dict]:
    lp = _nearest_contract(contracts, "put", -LONG_DELTA, spot, T, iv)
    if lp is None:
        return None
    premium = round(lp.ask_price, 2)
    if premium <= 0:
        return None
    return {
        "legs":    [{"action": "buy", "contract": lp}],
        "premium": premium,
    }


def _build_bull_call_spread(
    contracts: list[OptionContract],
    spot: float, T: float, iv: float,
) -> Optional[dict]:
    lc = _nearest_contract(contracts, "call", LONG_DELTA, spot, T, iv)
    if lc is None:
        return None
    wing_target = lc.strike + spot * SPREAD_WING_PCT
    calls_above = [c for c in contracts if c.right == "call" and c.strike > lc.strike and c.bid_price > 0]
    if not calls_above:
        return None
    sc = min(calls_above, key=lambda c: abs(c.strike - wing_target))
    debit = round(lc.ask_price - sc.bid_price, 2)
    if debit <= 0:
        return None
    return {
        "legs": [
            {"action": "buy",  "contract": lc},
            {"action": "sell", "contract": sc},
        ],
        "premium": debit,
    }


def _build_bear_put_spread(
    contracts: list[OptionContract],
    spot: float, T: float, iv: float,
) -> Optional[dict]:
    lp = _nearest_contract(contracts, "put", -LONG_DELTA, spot, T, iv)
    if lp is None:
        return None
    wing_target = lp.strike - spot * SPREAD_WING_PCT
    puts_below  = [c for c in contracts if c.right == "put" and c.strike < lp.strike and c.bid_price > 0]
    if not puts_below:
        return None
    sp = min(puts_below, key=lambda c: abs(c.strike - wing_target))
    debit = round(lp.ask_price - sp.bid_price, 2)
    if debit <= 0:
        return None
    return {
        "legs": [
            {"action": "buy",  "contract": lp},
            {"action": "sell", "contract": sp},
        ],
        "premium": debit,
    }


_BUILDERS = {
    "long_call":       _build_long_call,
    "long_put":        _build_long_put,
    "bull_call_spread": _build_bull_call_spread,
    "bear_put_spread":  _build_bear_put_spread,
}


# ── Main evaluator ─────────────────────────────────────────────────────────────

class EventTrader:
    """
    Runs the Monday evaluation:
      1. Gather catalyst tickers from scrapers.
      2. Evaluate each ticker for a long-gamma entry.
      3. Submit orders via Alpaca.
      4. Persist state.
    """

    def __init__(self, client: AlpacaClient):
        self.client = client

    def run_monday_evaluation(self, dry_run: bool = False):
        """
        Evaluate all candidate tickers and enter new positions.

        Parameters
        ----------
        dry_run : if True, compute signals and log intent but do NOT submit orders.
        """
        state       = _load_state()
        open_trades = state.get("open_trades", {})
        n_open      = len(open_trades)

        log.info("=== Monday evaluation %s | open=%d/%d ===",
                 date.today(), n_open, MAX_POSITIONS)

        if n_open >= MAX_POSITIONS:
            log.info("At max positions — skipping entry evaluation")
            return

        nav = self.client.account_value()
        log.info("Account NAV: $%.0f", nav)

        # Build prioritised ticker list: catalyst tickers first, then watchlist
        catalyst = _event_tickers()
        candidates = list(dict.fromkeys(catalyst + WATCHLIST))  # deduplicated, catalyst first

        entered = 0
        for ticker in candidates:
            if n_open + entered >= MAX_POSITIONS:
                break
            if ticker in open_trades:
                log.debug("[%s] already in open trades — skip", ticker)
                continue

            sig = compute_signal(
                ticker,
                iv_rank_max=IV_RANK_MAX,
                mom_window=20,
                trend_window=50,
                mom_min_pct=1.5,
            )
            if sig is None:
                continue

            spot      = sig["spot"]
            atm_iv    = sig["atm_iv"]
            structure = sig["structure"]
            direction = sig["direction"]

            # Fetch chain from Alpaca
            right_filter = "call" if direction == "CALL" else "put"
            chain = self.client.get_option_chain(
                ticker, spot,
                dte_min=DTE_MIN, dte_max=DTE_MAX,
                right=None,   # fetch both sides for spreads
            )
            if not chain:
                log.info("[%s] no option chain from Alpaca", ticker)
                continue

            contracts = _choose_expiry_contracts(chain)
            if not contracts:
                continue

            expiry = contracts[0].expiry
            T      = max((expiry - date.today()).days, 1) / 365.0

            builder = _BUILDERS.get(structure)
            if builder is None:
                continue
            info = builder(contracts, spot, T, atm_iv)
            if info is None:
                log.info("[%s] could not build legs for %s", ticker, structure)
                continue

            premium     = info["premium"]
            n_contracts = max(1, int(nav * MAX_PREMIUM_PCT / (premium * 100)))
            total_debit = round(premium * n_contracts * 100, 2)

            log.info("[%s] %s %s  x%d contracts  prem=%.2f  cost=$%.0f",
                     ticker, structure, direction, n_contracts, premium, total_debit)

            if dry_run:
                log.info("[%s] DRY RUN — skipping order submission", ticker)
                entered += 1
                continue

            # Submit: long legs first, then short legs (safety: never naked short)
            legs_ordered = sorted(info["legs"], key=lambda x: 0 if x["action"] == "buy" else 1)
            submitted_legs = []
            failed = False
            for leg in legs_ordered:
                action   = leg["action"]
                contract = leg["contract"]
                try:
                    if action == "buy":
                        oid = self.client.buy_to_open(contract.symbol, n_contracts)
                    else:
                        oid = self.client.sell_to_open(contract.symbol, n_contracts)
                    submitted_legs.append({
                        "action":  action,
                        "symbol":  contract.symbol,
                        "strike":  contract.strike,
                        "right":   contract.right,
                        "expiry":  contract.expiry.isoformat(),
                        "qty":     n_contracts,
                        "order_id": oid,
                    })
                except AlpacaOrderError as exc:
                    log.error("[%s] order failed: %s", ticker, exc)
                    failed = True
                    break

            if failed:
                # Attempt to unwind any legs already submitted
                for submitted in submitted_legs:
                    sym = submitted["symbol"]
                    qty = submitted["qty"]
                    try:
                        if submitted["action"] == "buy":
                            self.client.sell_to_close(sym, qty)
                        else:
                            self.client.buy_to_close(sym, qty)
                        log.info("[%s] unwound partial fill %s", ticker, sym)
                    except AlpacaOrderError:
                        log.error("[%s] UNWIND FAILED for %s — manual check needed", ticker, sym)
                continue

            # Persist to state
            open_trades[ticker] = {
                "trade_type":   structure,
                "direction":    direction,
                "legs":         submitted_legs,
                "premium":      premium,
                "n_contracts":  n_contracts,
                "open_date":    date.today().isoformat(),
                "expiry":       expiry.isoformat(),
                "nav_at_open":  nav,
            }
            entered += 1
            log.info("[%s] ✓ entered %s %s", ticker, structure, direction)

        state["open_trades"] = open_trades
        _save_state(state)
        log.info("Monday evaluation done — %d new position(s) entered", entered)
