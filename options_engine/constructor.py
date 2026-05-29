"""
Week 3 — Trade Constructor  (Alpaca-compliant)

Alpaca options levels supported:
  ✅ Covered Calls          — long stock + short OTM call
  ✅ Cash-Secured Puts      — cash reserve + short put
  ✅ Long Calls / Long Puts — outright long options
  ✅ Spreads                — bull-put, bear-call (vertical credit spreads)
  ✅ Covered Straddle       — long stock + short call + short put
  ✅ Multi-leg / Iron Condor— put credit spread + call credit spread
  ❌ Naked short call       — NOT allowed → auto-converts to bear-call spread
  ❌ Naked short put        — NOT allowed → auto-converts to bull-put spread
  ❌ Naked straddle/strangle→ NOT allowed → use iron condor or covered straddle

Design rules (short-premium PM playbook):
  - Short strikes : 16Δ by default (1σ OTM, ~84% single-wing POP)
  - Long strikes  : next liquid strike beyond a fixed wing width (IC only)
  - Sizing        : half-Kelly, capped at 5% of portfolio notional
  - Stop-loss     : auto-tagged at 2× credit received
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .pricer import ChainPricer, OptionLeg, bsm_delta


TradeType = Literal[
    "iron_condor",       # ✅ multi-leg spread
    "bull_put_spread",   # ✅ spread (cash-secured variant)
    "bear_call_spread",  # ✅ spread
    "cash_secured_put",  # ✅ cash-secured put
    "covered_call",      # ✅ covered call (requires shares)
    "long_straddle",     # ✅ long options
    "long_strangle",     # ✅ long options
]

# ── Alpaca compliance rules ───────────────────────────────────────────────────

ALPACA_ALLOWED: set[str] = {
    "iron_condor", "bull_put_spread", "bear_call_spread",
    "cash_secured_put", "covered_call", "long_straddle", "long_strangle",
}
ALPACA_FORBIDDEN: set[str] = {
    "naked_call", "naked_put", "naked_straddle", "naked_strangle",
}
# Forbidden → compliant replacement
ALPACA_FALLBACK: dict[str, str] = {
    "naked_call":     "bear_call_spread",
    "naked_put":      "bull_put_spread",
    "naked_straddle": "iron_condor",
    "naked_strangle": "iron_condor",
    "strangle":       "iron_condor",     # our old "strangle" was naked
    "straddle":       "iron_condor",     # our old "straddle" was naked
}


def alpaca_compliant(trade_type: str) -> str:
    """Map any trade type to its nearest Alpaca-compliant equivalent."""
    if trade_type in ALPACA_ALLOWED:
        return trade_type
    replacement = ALPACA_FALLBACK.get(trade_type, "iron_condor")
    import logging
    logging.getLogger(__name__).warning(
        "'%s' is not Alpaca-compliant → using '%s'", trade_type, replacement
    )
    return replacement


# ── Trade dataclass ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:           str
    trade_type:       TradeType
    legs:             list[OptionLeg]
    expiry:           str
    dte:              int
    spot:             float

    # P&L boundaries
    net_credit:       float   # premium collected per share (×100 per contract)
    max_profit:       float   # = net_credit (for IC); same for strangle before stop
    max_loss:         float   # IC: wing_width - credit; strangle: uses stop
    breakeven_upper:  float
    breakeven_lower:  float
    stop_loss_price:  float   # close if mark hits 2× credit (debit)

    # Aggregate Greeks (per 1 contract = 100 shares)
    net_delta:  float
    net_gamma:  float
    net_vega:   float
    net_theta:  float   # daily income in $

    # Sizing
    n_contracts:    int
    capital_at_risk: float   # max_loss × n_contracts × 100
    portfolio_pct:  float    # capital_at_risk / portfolio_value

    # Signal context (from Week 2)
    iv_rank:    float | None = None
    vrp:        float | None = None
    signal:     str   = "SELL"

    def summary(self) -> str:
        w = self.breakeven_upper - self.breakeven_lower
        return (
            f"\n{'─'*55}\n"
            f"  {self.trade_type.upper()}  {self.ticker}  exp={self.expiry}  DTE={self.dte}\n"
            f"  Spot: ${self.spot:.2f}\n"
            f"  Strikes: ${self.breakeven_lower:.1f} ── ${self.spot:.1f} ── ${self.breakeven_upper:.1f}\n"
            f"  Width: ${w:.1f}  |  Credit: ${self.net_credit:.2f}  |  Max loss: ${self.max_loss:.2f}\n"
            f"  Stop-loss mark: ${self.stop_loss_price:.2f} (2× credit)\n"
            f"  Greeks/contract: δ={self.net_delta:+.3f}  γ={self.net_gamma:.5f}"
            f"  ν={self.net_vega:.3f}  θ=${self.net_theta:.2f}/day\n"
            f"  Size: {self.n_contracts} contracts  |  Capital at risk: ${self.capital_at_risk:,.0f}"
            f"  ({self.portfolio_pct:.1f}% of portfolio)\n"
            f"  IV Rank: {self.iv_rank}  |  VRP: {self.vrp}%  |  Signal: {self.signal}\n"
            f"{'─'*55}"
        )


# ── Constructor ───────────────────────────────────────────────────────────────

class TradeConstructor:
    """
    Assembles iron condor or strangle from a priced chain.
    Reads go/no-go signal from VolSignal before constructing.
    """

    def __init__(
        self,
        portfolio_value: float = 100_000,
        max_portfolio_pct: float = 0.05,   # 5% max capital at risk per trade
        kelly_fraction: float = 0.5,       # half-Kelly
        stop_loss_mult: float = 2.0,       # close at 2× credit received
    ):
        self.portfolio_value    = portfolio_value
        self.max_portfolio_pct  = max_portfolio_pct
        self.kelly_fraction     = kelly_fraction
        self.stop_loss_mult     = stop_loss_mult

    # ── Public API ────────────────────────────────────────────────────────

    def iron_condor(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        short_delta: float = 0.16,
        wing_width:  float = 15.0,     # $ distance from short to long strike
        signal: dict | None = None,
    ) -> Trade:
        """
        Sell 16Δ call + 16Δ put, buy long wings width $ further OTM.
        Max loss is defined: wing_width - net_credit.
        """
        legs_short = pricer.find_strikes(chain, short_delta, -short_delta)
        sc = legs_short["short_call"]
        sp = legs_short["short_put"]

        # Long wings — find nearest strike ± wing_width
        lc = self._find_long_wing(chain, sc.strike + wing_width, "c")
        lp = self._find_long_wing(chain, sp.strike - wing_width, "p")

        lc.action, lp.action = "buy", "buy"

        net_credit = round((sc.mid + sp.mid) - (lc.mid + lp.mid), 2)
        wing       = round(lc.strike - sc.strike, 1)   # actual call spread width
        max_loss   = round(wing - net_credit, 2)

        return self._build_trade(
            ticker=pricer.ticker,
            trade_type="iron_condor",
            legs=[sc, lc, sp, lp],
            expiry=sc.expiry,
            spot=pricer.spot,
            net_credit=net_credit,
            max_loss=max_loss,
            be_upper=sc.strike + net_credit,
            be_lower=sp.strike - net_credit,
            signal=signal or {},
        )

    def bull_put_spread(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        short_delta: float = 0.30,   # more aggressive than IC for directional bias
        wing_width:  float = 10.0,
        signal: dict | None = None,
    ) -> Trade:
        """
        ✅ Alpaca: Spread (cash-secured variant).
        Sell OTM put + buy further OTM put. Profit if stock stays above short strike.
        Capital requirement = wing_width × 100 (held as cash).
        Use when: bullish bias + elevated IV.
        """
        sp = pricer.find_strikes(chain, 0.50, -short_delta)["short_put"]
        lp = self._find_long_wing(chain, sp.strike - wing_width, "p")
        lp.action = "buy"

        net_credit = round(sp.mid - lp.mid, 2)
        wing       = round(sp.strike - lp.strike, 1)
        max_loss   = round(wing - net_credit, 2)

        return self._build_trade(
            ticker=pricer.ticker, trade_type="bull_put_spread",
            legs=[sp, lp], expiry=sp.expiry, spot=pricer.spot,
            net_credit=net_credit, max_loss=max_loss,
            be_upper=pricer.spot * 999,   # unlimited upside
            be_lower=sp.strike - net_credit,
            signal=signal or {},
        )

    def bear_call_spread(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        short_delta: float = 0.30,
        wing_width:  float = 10.0,
        signal: dict | None = None,
    ) -> Trade:
        """
        ✅ Alpaca: Spread.
        Sell OTM call + buy further OTM call. Profit if stock stays below short strike.
        Max loss = wing_width - credit (fully defined).
        Use when: bearish/neutral bias + elevated IV.
        """
        sc = pricer.find_strikes(chain, short_delta, -0.50)["short_call"]
        lc = self._find_long_wing(chain, sc.strike + wing_width, "c")
        lc.action = "buy"

        net_credit = round(sc.mid - lc.mid, 2)
        wing       = round(lc.strike - sc.strike, 1)
        max_loss   = round(wing - net_credit, 2)

        return self._build_trade(
            ticker=pricer.ticker, trade_type="bear_call_spread",
            legs=[sc, lc], expiry=sc.expiry, spot=pricer.spot,
            net_credit=net_credit, max_loss=max_loss,
            be_upper=sc.strike + net_credit,
            be_lower=0,  # unlimited downside protection
            signal=signal or {},
        )

    def cash_secured_put(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        short_delta: float = 0.30,
        signal: dict | None = None,
    ) -> Trade:
        """
        ✅ Alpaca: Cash-Secured Put.
        Sell OTM put; broker holds cash = strike × 100 as collateral.
        Profit = full credit if stock stays above strike at expiry.
        Use when: willing to own stock at strike price, IV is elevated.
        """
        sp = pricer.find_strikes(chain, 0.50, -short_delta)["short_put"]
        sp.action = "sell"

        net_credit = round(sp.mid, 2)
        # Max loss = strike - credit (stock goes to zero; practical stop = 3× credit)
        max_loss = round(sp.strike - net_credit, 2)
        # Capital requirement (cash held): strike × 100 per contract
        cash_req = round(sp.strike * 100, 2)

        return self._build_trade(
            ticker=pricer.ticker, trade_type="cash_secured_put",
            legs=[sp], expiry=sp.expiry, spot=pricer.spot,
            net_credit=net_credit, max_loss=net_credit * 3,   # use 3× for sizing
            be_upper=pricer.spot * 999,
            be_lower=sp.strike - net_credit,
            signal=signal or {},
            extra={"cash_requirement": cash_req},
        )

    def covered_call(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        short_delta: float = 0.30,
        signal: dict | None = None,
    ) -> Trade:
        """
        ✅ Alpaca: Covered Call.
        Sell OTM call against 100 long shares already held.
        Caps upside at short strike; premium reduces cost basis.
        Use when: long stock holder wants income in range-bound market.
        NOTE: Requires existing 100-share position per contract.
        """
        sc = pricer.find_strikes(chain, short_delta, -0.50)["short_call"]
        sc.action = "sell"

        net_credit = round(sc.mid, 2)
        # Max loss on call leg = unlimited (but offset by long stock gains)
        # For sizing: use credit × 3 as practical max loss on the option leg
        max_loss = round(net_credit * 3, 2)

        return self._build_trade(
            ticker=pricer.ticker, trade_type="covered_call",
            legs=[sc], expiry=sc.expiry, spot=pricer.spot,
            net_credit=net_credit, max_loss=max_loss,
            be_upper=sc.strike + net_credit,
            be_lower=pricer.spot - net_credit,   # cost-basis reduction
            signal=signal or {},
            extra={"requires_shares": 100},
        )

    def long_straddle(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        signal: dict | None = None,
    ) -> Trade:
        """
        ✅ Alpaca: Long options.
        Buy ATM call + ATM put. Profit from large move in either direction.
        Use when: low IV rank, binary event expected (earnings, FDA).
        Max loss = total debit paid.
        """
        atm_call = self._atm_leg(chain, "c", pricer.spot)
        atm_put  = self._atm_leg(chain, "p", pricer.spot)
        atm_call.action = atm_put.action = "buy"

        net_debit = round(atm_call.mid + atm_put.mid, 2)
        return self._build_trade(
            ticker=pricer.ticker, trade_type="long_straddle",
            legs=[atm_call, atm_put], expiry=atm_call.expiry, spot=pricer.spot,
            net_credit=-net_debit,   # negative = debit paid
            max_loss=net_debit,
            be_upper=atm_call.strike + net_debit,
            be_lower=atm_put.strike  - net_debit,
            signal=signal or {},
        )

    def long_strangle(
        self,
        chain: pd.DataFrame,
        pricer: ChainPricer,
        long_delta: float = 0.30,
        signal: dict | None = None,
    ) -> Trade:
        """
        ✅ Alpaca: Long options.
        Buy OTM call + OTM put. Cheaper than straddle, needs bigger move to profit.
        Use when: low IV rank, expecting large move but not sure of direction.
        """
        lc = pricer.find_strikes(chain, long_delta, -long_delta)["short_call"]
        lp = pricer.find_strikes(chain, long_delta, -long_delta)["short_put"]
        lc.action = lp.action = "buy"

        net_debit = round(lc.mid + lp.mid, 2)
        return self._build_trade(
            ticker=pricer.ticker, trade_type="long_strangle",
            legs=[lc, lp], expiry=lc.expiry, spot=pricer.spot,
            net_credit=-net_debit,
            max_loss=net_debit,
            be_upper=lc.strike + net_debit,
            be_lower=lp.strike  - net_debit,
            signal=signal or {},
        )

    # ── Kelly sizing ──────────────────────────────────────────────────────

    def kelly_size(self, pop: float, max_profit: float, max_loss: float) -> int:
        """
        Half-Kelly contract count, capped at max_portfolio_pct of capital.
        f* = (p·b - q) / b   where b = max_profit / max_loss
        Returns number of contracts (minimum 1).
        """
        if max_loss <= 0 or max_profit <= 0:
            return 1
        p = pop / 100
        q = 1 - p
        b = max_profit / max_loss
        f_full = (p * b - q) / b
        f_half = max(0.0, f_full * self.kelly_fraction)

        # Capital at risk per contract = max_loss × 100 shares
        capital_per_contract = max_loss * 100
        max_contracts_by_pct = int(
            (self.portfolio_value * self.max_portfolio_pct) / capital_per_contract
        )
        kelly_contracts = int(f_half * self.portfolio_value / capital_per_contract)
        return max(1, min(kelly_contracts, max_contracts_by_pct))

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_trade(
        self, ticker, trade_type, legs, expiry, spot,
        net_credit, max_loss, be_upper, be_lower, signal,
        extra: dict | None = None,
    ) -> Trade:
        dte = self._dte(expiry)

        # Aggregate Greeks (sells are negative contribution, buys positive)
        sign = {"sell": -1, "buy": 1}
        def agg(attr):
            return round(sum(sign[l.action] * getattr(l, attr) for l in legs), 5)

        # Theta in $ per day per contract (×100 shares)
        net_theta_dollar = round(agg("theta") * 100, 2)

        # Placeholder POP for sizing — will be overwritten by Week 4 MC engine
        pop_placeholder = 68.0   # rough: 16Δ wings → ~68% POP

        n = self.kelly_size(pop_placeholder, net_credit, max_loss)
        car = round(max_loss * n * 100, 2)

        trade = Trade(
            ticker=ticker, trade_type=trade_type, legs=legs, expiry=expiry,
            dte=dte, spot=spot, net_credit=net_credit, max_profit=net_credit,
            max_loss=max_loss,
            breakeven_upper=round(be_upper, 2),
            breakeven_lower=round(be_lower, 2),
            stop_loss_price=round(abs(net_credit) * self.stop_loss_mult, 2),
            net_delta=agg("delta"), net_gamma=agg("gamma"),
            net_vega=agg("vega"), net_theta=net_theta_dollar,
            n_contracts=n, capital_at_risk=car,
            portfolio_pct=round(car / self.portfolio_value * 100, 2),
            iv_rank=signal.get("iv_rank"), vrp=signal.get("vrp"),
            signal=signal.get("signal", "SELL"),
        )
        # Attach extra metadata (cash_requirement, requires_shares, etc.)
        if extra:
            trade._extra = extra
        return trade

    def _find_long_wing(self, chain: pd.DataFrame, target_strike: float, flag: str) -> OptionLeg:
        df = chain[chain["flag"] == flag].dropna(subset=["mid"])
        idx = (df["strike"] - target_strike).abs().idxmin()
        r = df.loc[idx]
        return OptionLeg(
            ticker="", expiry=r["expiry"], strike=r["strike"], flag=flag, action="buy",
            delta=r["delta"], gamma=r["gamma"], vega=r["vega"], theta=r["theta"],
            iv=r["iv"], mid=r["mid"],
        )

    def _atm_leg(self, chain: pd.DataFrame, flag: str, spot: float) -> OptionLeg:
        df = chain[chain["flag"] == flag].dropna(subset=["mid"])
        idx = (df["strike"] - spot).abs().idxmin()
        r = df.loc[idx]
        return OptionLeg(
            ticker="", expiry=r["expiry"], strike=r["strike"], flag=flag, action="sell",
            delta=r["delta"], gamma=r["gamma"], vega=r["vega"], theta=r["theta"],
            iv=r["iv"], mid=r["mid"],
        )

    @staticmethod
    def _dte(expiry: str) -> int:
        from datetime import date
        return (date.fromisoformat(expiry) - date.today()).days
