"""
Alpaca paper-trading client wrapper.

Responsibilities
----------------
- Load credentials from env vars (never hardcoded).
- Fetch option chains for a given underlying and DTE window.
- Submit market orders for long calls/puts and debit spreads.
- Query account value and open option positions.

All order submissions are best-effort: the function logs the order and
raises AlpacaOrderError on failure so the caller can skip the ticker.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetStatus, ContractType, OrderSide, TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    GetOrdersRequest,
    MarketOrderRequest,
)

log = logging.getLogger(__name__)

# ── Custom exceptions ──────────────────────────────────────────────────────────

class AlpacaClientError(RuntimeError):
    """Raised when credentials are missing or the client cannot be initialised."""

class AlpacaOrderError(RuntimeError):
    """Raised when an order submission is rejected by Alpaca."""


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    """Thin wrapper around Alpaca's option contract object."""
    symbol:      str          # OCC-style  e.g. "SPY240119C00480000"
    underlying:  str          # "SPY"
    strike:      float
    expiry:      date
    right:       str          # "call" | "put"
    ask_price:   float        # last ask (may be 0 if market closed)
    bid_price:   float        # last bid
    iv:          float        # implied vol from Alpaca, or 0.0 if unavailable

    @property
    def mid(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return max(self.bid_price, self.ask_price)


@dataclass
class OpenPosition:
    """A live option position held in the account."""
    ticker:       str
    trade_type:   str          # "long_call" | "long_put" | "bull_call_spread" | "bear_put_spread"
    direction:    str          # "CALL" | "PUT"
    legs:         list[dict]   # [{"action": "buy"|"sell", "symbol": <OCC>, "qty": int}, ...]
    premium:      float        # net debit paid per share (× 100 = per contract)
    n_contracts:  int
    open_date:    date
    expiry:       date
    extra:        dict = field(default_factory=dict)


# ── Client ─────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """
    Thin wrapper around alpaca-py TradingClient.

    Credentials are read from environment variables:
        ALPACA_API_KEY     — paper or live key
        ALPACA_SECRET_KEY  — corresponding secret
        ALPACA_PAPER       — "true" (default) / "false"
    """

    # Options require a slightly wider strike range to guarantee a hit
    _STRIKE_BAND_PCT = 0.35   # ±35 % of spot

    def __init__(self):
        key    = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        paper  = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

        if not key or not secret:
            raise AlpacaClientError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment "
                "variables (or in a .env file loaded before this import). "
                "Never hardcode credentials."
            )

        self._client = TradingClient(api_key=key, secret_key=secret, paper=paper)
        mode = "paper" if paper else "LIVE"
        log.info("AlpacaClient ready — %s account", mode)

    # ── Account ────────────────────────────────────────────────────────────────

    def account_value(self) -> float:
        """Return total portfolio value in USD."""
        acct = self._client.get_account()
        return float(acct.portfolio_value)

    # ── Option chain ───────────────────────────────────────────────────────────

    def get_option_chain(
        self,
        ticker:     str,
        spot:       float,
        dte_min:    int = 15,
        dte_max:    int = 45,
        right:      Optional[str] = None,   # "call" | "put" | None (both)
    ) -> list[OptionContract]:
        """
        Return option contracts for *ticker* in the requested DTE window.

        Parameters
        ----------
        ticker  : underlying symbol
        spot    : current stock price (used to bound strike range)
        dte_min : minimum days to expiration
        dte_max : maximum days to expiration
        right   : "call", "put", or None for both

        Returns
        -------
        List of OptionContract objects sorted by (expiry, strike).
        """
        today    = date.today()
        exp_lo   = today + timedelta(days=dte_min)
        exp_hi   = today + timedelta(days=dte_max)
        strike_lo = spot * (1 - self._STRIKE_BAND_PCT)
        strike_hi = spot * (1 + self._STRIKE_BAND_PCT)

        contract_type = None
        if right == "call":
            contract_type = ContractType.CALL
        elif right == "put":
            contract_type = ContractType.PUT

        req = GetOptionContractsRequest(
            underlying_symbols=[ticker],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=exp_lo.isoformat(),
            expiration_date_lte=exp_hi.isoformat(),
            strike_price_gte=str(round(strike_lo, 2)),
            strike_price_lte=str(round(strike_hi, 2)),
            type=contract_type,
            limit=500,
        )

        try:
            resp = self._client.get_option_contracts(req)
            raw  = resp.option_contracts if hasattr(resp, "option_contracts") else []
        except Exception as exc:
            log.warning("[%s] chain fetch failed: %s", ticker, exc)
            return []

        contracts = []
        for c in raw:
            try:
                ask = float(c.ask_price or 0)
                bid = float(c.bid_price or 0)
                iv  = float(c.implied_volatility or 0) if hasattr(c, "implied_volatility") else 0.0
                contracts.append(OptionContract(
                    symbol=str(c.symbol),
                    underlying=ticker,
                    strike=float(c.strike_price),
                    expiry=c.expiration_date if isinstance(c.expiration_date, date)
                           else date.fromisoformat(str(c.expiration_date)),
                    right=c.type.value if hasattr(c.type, "value") else str(c.type),
                    ask_price=ask,
                    bid_price=bid,
                    iv=iv,
                ))
            except Exception as exc:
                log.debug("Skipping contract parse error: %s", exc)

        contracts.sort(key=lambda x: (x.expiry, x.strike))
        log.debug("[%s] chain: %d contracts (DTE %d–%d)", ticker, len(contracts), dte_min, dte_max)
        return contracts

    # ── Order helpers ──────────────────────────────────────────────────────────

    def _market_order(self, symbol: str, qty: int, side: OrderSide) -> str:
        """Submit a single leg market order; return order id."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = self._client.submit_order(req)
            log.info("  order %s  %s %s x%d  → id=%s",
                     side.value.upper(), symbol, side.value, abs(qty), order.id)
            return str(order.id)
        except Exception as exc:
            raise AlpacaOrderError(f"Order failed {side.value} {symbol} x{qty}: {exc}") from exc

    def buy_to_open(self, symbol: str, qty: int) -> str:
        return self._market_order(symbol, qty, OrderSide.BUY)

    def sell_to_open(self, symbol: str, qty: int) -> str:
        return self._market_order(symbol, qty, OrderSide.SELL)

    def buy_to_close(self, symbol: str, qty: int) -> str:
        return self._market_order(symbol, qty, OrderSide.BUY)

    def sell_to_close(self, symbol: str, qty: int) -> str:
        return self._market_order(symbol, qty, OrderSide.SELL)

    # ── Position queries ───────────────────────────────────────────────────────

    def get_open_positions(self) -> list:
        """Return all open positions (Alpaca Position objects)."""
        try:
            return self._client.get_all_positions()
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
            return []

    def get_option_position_value(self, occ_symbol: str) -> Optional[float]:
        """
        Return current market value of one option position, or None if not found.
        Value is per-contract (already × 100 by Alpaca).
        """
        try:
            pos = self._client.get_open_position(occ_symbol)
            return float(pos.market_value)
        except Exception:
            return None

    def close_position(self, occ_symbol: str):
        """Close an option position by OCC symbol (Alpaca handles side detection)."""
        try:
            self._client.close_position(occ_symbol)
            log.info("  closed position %s", occ_symbol)
        except Exception as exc:
            raise AlpacaOrderError(f"Close failed for {occ_symbol}: {exc}") from exc
