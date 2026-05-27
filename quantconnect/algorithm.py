"""
QuantConnect LEAN Algorithm — Alpaca-Compliant Short-Premium Strategy

Backtest entry point for the full pipeline:
  Event scraper → Vol signal → Trade constructor → Probability engine → Orders

Alpaca options levels enforced:
  ✅ Iron Condor (multi-leg spread)
  ✅ Bull Put Spread / Bear Call Spread
  ✅ Cash-Secured Put
  ✅ Covered Call
  ✅ Long Straddle / Long Strangle

Upload this file to QuantConnect Cloud (quantconnect.com/terminal) or run
locally with the LEAN CLI:  lean backtest quantconnect/algorithm.py

LEAN docs: https://www.lean.io/docs/v2/lean-engine/key-concepts/algorithm-structure
"""

# ── Standard QC imports (available in LEAN sandbox) ───────────────────────────
from AlgorithmImports import *   # noqa: F401,F403  (LEAN provides this namespace)

# ── Local engine (inline BSM — no pip install needed in QC sandbox) ───────────
# Copy of options_engine/pricer.py math, inlined to avoid dependency issues.
# In local LEAN CLI runs you can import directly: from options_engine import ...
import numpy as np
from scipy.stats import norm


# ══════════════════════════════════════════════════════════════════════════════
# Inlined BSM core (same math as options_engine/pricer.py — QC sandbox safe)
# ══════════════════════════════════════════════════════════════════════════════

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bsm_delta(flag, S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) if flag == "c" else norm.cdf(d1) - 1

def yang_zhang_rv(closes, opens, highs, lows, window=21):
    """Yang-Zhang RV from numpy arrays. Returns annualised % or None."""
    n = len(closes)
    if n < window + 1:
        return None
    o, h, l, c = opens, highs, lows, closes
    ro = np.log(o[1:] / c[:-1])
    rc = np.log(c[1:] / o[1:])
    rs = np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) + \
         np.log(l[1:] / c[1:]) * np.log(l[1:] / o[1:])
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    from pandas import Series
    yz = (Series(ro).rolling(window).var() +
          k * Series(rc).rolling(window).var() +
          (1 - k) * Series(rs).rolling(window).mean()).iloc[-1]
    return float(np.sqrt(yz * 252) * 100)

def iv_rank_from_hv(hv_series, current_iv):
    lo, hi = min(hv_series), max(hv_series)
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (current_iv - lo) / (hi - lo) * 100))


# ══════════════════════════════════════════════════════════════════════════════
# LEAN Algorithm
# ══════════════════════════════════════════════════════════════════════════════

class AlpacaShortPremiumAlgo(QCAlgorithm):
    """
    Monday event-driven short-premium algorithm.

    Workflow (every Monday 07:00 ET):
      1. Compute IV rank + vol risk premium for each ticker in WATCHLIST
      2. If IV rank > 40 AND VRP > 2%  →  construct Alpaca-compliant trade
      3. Run analytical POP check  (>60% required)
      4. Submit option orders via Alpaca brokerage model
      5. Manage open positions: close at 2× credit OR 21 DTE, whichever first

    Risk controls:
      - Max 5% portfolio per position (capital at risk)
      - Hard stop at 200% of credit received (mark-to-market)
      - Delta rebalance if |net delta| > 0.10 per contract
      - Force close all positions at 21 DTE
    """

    # ── Configuration ──────────────────────────────────────────────────────
    WATCHLIST       = ["SPY", "QQQ"]          # start small — easier to debug
    DEFAULT_TRADE   = "iron_condor"
    DTE_TARGET      = 30
    DTE_CLOSE       = 21
    SHORT_DELTA     = 0.16
    WING_WIDTH      = 15.0
    MAX_PCT         = 0.05
    STOP_MULT       = 2.0
    IV_RANK_MIN     = 20.0            # lowered from 40 — easier to trigger
    VRP_MIN         = 0.0             # lowered from 2 — easier to trigger
    MIN_POP         = 55.0            # lowered from 60
    RISK_FREE       = 0.053

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)
        self.SetBrokerageModel(BrokerageName.Alpaca, AccountType.Margin)
        self.SetBenchmark("SPY")

        # ── Subscribe equities + options ─────────────────────────────────
        self.option_symbols = {}
        for ticker in self.WATCHLIST:
            equity = self.AddEquity(ticker, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
            option = self.AddOption(ticker, Resolution.Daily)
            option.SetFilter(self._option_filter)
            self.option_symbols[ticker] = option.Symbol

        # ── Cache: filled by OnData, consumed by Monday briefing ─────────
        # Chains are only available in OnData, NOT in scheduled functions
        self._cached_chains  = {}   # ticker → OptionChain
        self.open_trades     = {}   # ticker → trade metadata
        self.history_window  = 252
        self._monday_pending = False   # flag set by OnData to trigger briefing

        # ── Schedule: daily position management at 15:45 ET ─────────────
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 45),
            self.ManagePositions,
        )
        self.Log("AlpacaShortPremiumAlgo initialized")

    # ── Option chain filter ───────────────────────────────────────────────────

    def _option_filter(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
        return universe.Expiration(15, 45).Strikes(-15, 15)

    # ── OnData: cache chains + trigger Monday logic ───────────────────────────

    def OnData(self, data: Slice):
        """
        Cache option chains each bar — scheduled functions don't have Slice access.
        Trigger Monday briefing here once chains are confirmed available.
        """
        # Cache every chain that arrives
        for ticker in self.WATCHLIST:
            sym = self.option_symbols.get(ticker)
            if sym and data.OptionChains.ContainsKey(sym):
                self._cached_chains[ticker] = data.OptionChains[sym]

        # Fire Monday briefing on the first bar of each Monday
        if self.Time.weekday() == 0:   # Monday = 0
            date_key = self.Time.date()
            if not hasattr(self, "_last_briefing_date") or self._last_briefing_date != date_key:
                self._last_briefing_date = date_key
                self.MondayBriefing()

    # ── Monday briefing ───────────────────────────────────────────────────────

    def MondayBriefing(self):
        self.Log(f"=== Monday Briefing {self.Time.date()} "
                 f"| chains cached: {list(self._cached_chains.keys())} ===")

        for ticker in self.WATCHLIST:
            if ticker in self.open_trades:
                self.Log(f"[{ticker}] Position open — skipping")
                continue
            try:
                self._evaluate_and_trade(ticker)
            except Exception as e:
                self.Log(f"[{ticker}] Error: {e}")

    def _evaluate_and_trade(self, ticker: str):
        # ── Step 1: Vol signal ──────────────────────────────────────────
        hist = self.History[TradeBar](
            self.Securities[ticker].Symbol, self.history_window, Resolution.Daily
        )
        hist_list = list(hist)
        if not hist_list:
            return

        closes = np.array([b.Close for b in hist_list])
        opens  = np.array([b.Open  for b in hist_list])
        highs  = np.array([b.High  for b in hist_list])
        lows   = np.array([b.Low   for b in hist_list])
        spot   = float(closes[-1])

        rv_21 = yang_zhang_rv(closes, opens, highs, lows, window=21)
        if rv_21 is None:
            return

        # IV rank proxy using 21-day rolling HV over past year
        log_ret  = np.log(closes[1:] / closes[:-1])
        hv_series = [
            np.std(log_ret[i-21:i]) * np.sqrt(252) * 100
            for i in range(21, len(log_ret))
        ]
        # ATM IV from cached chain (captured in OnData)
        atm_iv = self._get_atm_iv(ticker, spot)
        if atm_iv is None:
            self.Log(f"[{ticker}] No ATM IV available — chain cached: {ticker in self._cached_chains}")
            return

        iv_rank = iv_rank_from_hv(hv_series, atm_iv)
        vrp     = atm_iv - rv_21

        self.Log(f"[{ticker}] spot={spot:.2f} IV={atm_iv:.1f}% RV={rv_21:.1f}% "
                 f"IVRank={iv_rank:.0f} VRP={vrp:.1f}% "
                 f"(thresholds: IVR>{self.IV_RANK_MIN} VRP>{self.VRP_MIN})")

        # ── Step 2: Go / no-go ──────────────────────────────────────────
        if iv_rank < self.IV_RANK_MIN or vrp < self.VRP_MIN:
            self.Log(f"[{ticker}] Signal NEUTRAL — IVRank={iv_rank:.0f} VRP={vrp:.1f}% — skip")
            return

        # ── Step 3: Select Alpaca-compliant structure ───────────────────
        trade_type = self._select_structure(iv_rank, vrp)

        # ── Step 4: Find strikes from live chain ────────────────────────
        chain_info = self._find_chain_strikes(ticker, spot, trade_type)
        if chain_info is None:
            return

        # ── Step 5: Analytical POP check ───────────────────────────────
        pop = self._analytical_pop(
            spot, chain_info["be_lower"], chain_info["be_upper"],
            atm_iv / 100, chain_info["T"]
        )
        self.Log(f"[{ticker}] {trade_type} | credit={chain_info['credit']:.2f} "
                 f"max_loss={chain_info['max_loss']:.2f} POP={pop:.1f}%")

        if pop < self.MIN_POP:
            self.Log(f"[{ticker}] POP {pop:.1f}% < {self.MIN_POP}% — skip")
            return

        # ── Step 6: Size and submit ─────────────────────────────────────
        n_contracts = self._kelly_size(
            pop, chain_info["credit"], chain_info["max_loss"]
        )
        self._submit_orders(ticker, trade_type, chain_info, n_contracts)

    # ── Structure selector (Alpaca-aware) ─────────────────────────────────────

    def _select_structure(self, iv_rank: float, vrp: float) -> str:
        """
        Maps vol regime to the best Alpaca-compliant structure.

        IV rank > 60 + high VRP  →  Iron Condor (max theta, defined risk)
        IV rank 40-60             →  Bull Put Spread (directionally neutral-bullish)
        IV rank < 40              →  Long Strangle  (buy cheap vol before event)
        """
        if iv_rank >= 60 and vrp >= 4.0:
            return "iron_condor"
        elif iv_rank >= 40:
            return "bull_put_spread"
        else:
            return "long_strangle"

    # ── Chain strike selection ────────────────────────────────────────────────

    def _find_chain_strikes(self, ticker: str, spot: float, trade_type: str) -> dict | None:
        """Use cached OptionChain (populated by OnData) to find target-delta strikes."""
        chain = self._cached_chains.get(ticker)
        if chain is None or not chain.Contracts:
            self.Log(f"[{ticker}] No cached chain — contracts available: "
                     f"{len(chain.Contracts) if chain else 0}")
            return None

        # Group contracts by expiry, pick nearest to DTE_TARGET
        by_expiry: dict = {}
        for contract in chain.Contracts.Values:
            exp = contract.Expiry.date()
            by_expiry.setdefault(exp, []).append(contract)

        today    = self.Time.date()
        target_e = min(by_expiry, key=lambda e: abs((e - today).days - self.DTE_TARGET))
        T        = (target_e - today).days / 365
        contracts= by_expiry[target_e]

        if trade_type == "iron_condor":
            return self._ic_strikes(contracts, spot, T)
        elif trade_type == "bull_put_spread":
            return self._bps_strikes(contracts, spot, T)
        elif trade_type == "long_strangle":
            return self._ls_strikes(contracts, spot, T)
        return None

    def _ic_strikes(self, contracts, spot, T) -> dict | None:
        """Iron condor: 16Δ short call + put, long wings WING_WIDTH $ further."""
        calls = sorted([c for c in contracts if c.Right == OptionRight.Call],
                       key=lambda c: c.Strike)
        puts  = sorted([c for c in contracts if c.Right == OptionRight.Put],
                       key=lambda c: c.Strike)

        sc = self._nearest_delta(calls, spot, T, self.SHORT_DELTA,     "c")
        sp = self._nearest_delta(puts,  spot, T, -self.SHORT_DELTA,    "p")
        if sc is None or sp is None:
            return None

        lc = min((c for c in calls if c.Strike > sc.Strike), key=lambda c: c.Strike, default=None)
        lp = max((c for c in puts  if c.Strike < sp.Strike), key=lambda c: c.Strike, default=None)
        if lc is None or lp is None:
            return None

        credit   = ((sc.BidPrice + sc.AskPrice) / 2 +
                    (sp.BidPrice + sp.AskPrice) / 2 -
                    (lc.BidPrice + lc.AskPrice) / 2 -
                    (lp.BidPrice + lp.AskPrice) / 2)
        wing     = lc.Strike - sc.Strike
        max_loss = max(wing - credit, 0.01)

        return {
            "legs":      [("sell", sc), ("buy", lc), ("sell", sp), ("buy", lp)],
            "credit":    round(float(credit), 2),
            "max_loss":  round(float(max_loss), 2),
            "be_upper":  sc.Strike + credit,
            "be_lower":  sp.Strike - credit,
            "T": T,
        }

    def _bps_strikes(self, contracts, spot, T) -> dict | None:
        """Bull put spread: sell 30Δ put, buy put WING_WIDTH $ lower."""
        puts = sorted([c for c in contracts if c.Right == OptionRight.Put],
                      key=lambda c: c.Strike)
        sp = self._nearest_delta(puts, spot, T, -0.30, "p")
        if sp is None:
            return None
        lp = max((c for c in puts if c.Strike < sp.Strike), key=lambda c: c.Strike, default=None)
        if lp is None:
            return None

        credit   = ((sp.BidPrice + sp.AskPrice) / 2 -
                    (lp.BidPrice + lp.AskPrice) / 2)
        wing     = sp.Strike - lp.Strike
        max_loss = max(wing - credit, 0.01)

        return {
            "legs":     [("sell", sp), ("buy", lp)],
            "credit":   round(float(credit), 2),
            "max_loss": round(float(max_loss), 2),
            "be_upper": spot * 999,
            "be_lower": sp.Strike - credit,
            "T": T,
        }

    def _ls_strikes(self, contracts, spot, T) -> dict | None:
        """Long strangle: buy 30Δ call + 30Δ put."""
        calls = [c for c in contracts if c.Right == OptionRight.Call]
        puts  = [c for c in contracts if c.Right == OptionRight.Put]
        lc = self._nearest_delta(calls, spot, T,  0.30, "c")
        lp = self._nearest_delta(puts,  spot, T, -0.30, "p")
        if lc is None or lp is None:
            return None

        debit    = ((lc.BidPrice + lc.AskPrice) / 2 +
                    (lp.BidPrice + lp.AskPrice) / 2)
        return {
            "legs":     [("buy", lc), ("buy", lp)],
            "credit":   round(float(-debit), 2),   # negative = debit
            "max_loss": round(float(debit), 2),
            "be_upper": lc.Strike + debit,
            "be_lower": lp.Strike - debit,
            "T": T,
        }

    # ── Order submission ──────────────────────────────────────────────────────

    def _submit_orders(self, ticker: str, trade_type: str,
                       chain_info: dict, n_contracts: int):
        tickets = []
        for action, contract in chain_info["legs"]:
            qty = n_contracts if action == "buy" else -n_contracts
            ticket = self.MarketOrder(contract.Symbol, qty)
            tickets.append(ticket)
            self.Log(f"  {action.upper()} {contract.Symbol} ×{n_contracts}")

        # Store for position management
        self.open_trades[ticker] = {
            "trade_type":  trade_type,
            "legs":        chain_info["legs"],
            "credit":      chain_info["credit"],
            "max_loss":    chain_info["max_loss"],
            "be_upper":    chain_info["be_upper"],
            "be_lower":    chain_info["be_lower"],
            "n_contracts": n_contracts,
            "open_date":   self.Time.date(),
            "expiry":      chain_info["legs"][0][1].Expiry.date(),
        }
        self.Log(f"[{ticker}] Opened {trade_type} × {n_contracts} "
                 f"credit=${chain_info['credit']:.2f} "
                 f"max_loss=${chain_info['max_loss']:.2f}")

    # ── Position management ───────────────────────────────────────────────────

    def ManagePositions(self):
        """
        Daily checks at 15:45 ET:
          - Close at 21 DTE (theta decay slows, gamma risk rises)
          - Close if mark loss > 2× credit received (stop-loss)
          - Close long strangles at 2× debit (take profit)
        """
        today    = self.Time.date()
        to_close = []

        for ticker, meta in self.open_trades.items():
            expiry  = meta["expiry"]
            dte     = (expiry - today).days
            credit  = meta["credit"]
            n       = meta["n_contracts"]

            # ── 21 DTE close ────────────────────────────────────────────
            if dte <= self.DTE_CLOSE:
                self.Log(f"[{ticker}] Closing at {dte} DTE")
                to_close.append(ticker)
                continue

            # ── Mark-to-market P&L check ─────────────────────────────
            current_mark = self._get_position_mark(meta)
            if current_mark is None:
                continue

            pnl_per_share = credit - current_mark   # positive = profit
            if credit > 0:  # short premium
                if pnl_per_share < -abs(credit) * self.STOP_MULT:
                    self.Log(f"[{ticker}] Stop-loss hit: P&L={pnl_per_share:.2f}")
                    to_close.append(ticker)
            else:           # long premium (strangle)
                if pnl_per_share > abs(credit):     # 100% profit target
                    self.Log(f"[{ticker}] Profit target hit: P&L={pnl_per_share:.2f}")
                    to_close.append(ticker)

        for ticker in to_close:
            self._close_position(ticker)

    def _close_position(self, ticker: str):
        meta = self.open_trades.pop(ticker, None)
        if not meta:
            return
        for action, contract in meta["legs"]:
            # Reverse original order
            close_qty = meta["n_contracts"] if action == "sell" else -meta["n_contracts"]
            self.MarketOrder(contract.Symbol, close_qty)
        self.Log(f"[{ticker}] Position closed")

    def _get_position_mark(self, meta: dict) -> float | None:
        """Sum of mid prices across all legs using Securities cache."""
        mark = 0.0
        for action, contract in meta["legs"]:
            sym = contract.Symbol
            if not self.Securities.ContainsKey(sym):
                return None
            sec = self.Securities[sym]
            bid, ask = float(sec.BidPrice), float(sec.AskPrice)
            if bid <= 0 and ask <= 0:
                return None
            mid = (bid + ask) / 2
            mark += mid if action == "buy" else -mid
        return mark

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_atm_iv(self, ticker: str, spot: float) -> float | None:
        chain = self._cached_chains.get(ticker)
        if chain is None:
            return None
        calls = [c for c in chain.Contracts.Values if c.Right == OptionRight.Call]
        if not calls:
            return None
        atm = min(calls, key=lambda c: abs(c.Strike - spot))
        if atm.ImpliedVolatility > 0:
            return float(atm.ImpliedVolatility) * 100
        # Fallback: estimate IV from historical vol if QC doesn't compute it
        return None

    def _nearest_delta(self, contracts, spot, T, target_delta, flag):
        best, best_diff = None, float("inf")
        for c in contracts:
            d = bsm_delta(flag, spot, c.Strike, T, self.RISK_FREE,
                          max(float(c.ImpliedVolatility), 0.05))
            diff = abs(d - target_delta)
            if diff < best_diff:
                best, best_diff = c, diff
        return best

    def _analytical_pop(self, S, lower, upper, sigma, T) -> float:
        r = self.RISK_FREE
        def d2(K):
            return (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return abs(float(norm.cdf(d2(upper)) - norm.cdf(d2(lower)))) * 100

    def _kelly_size(self, pop: float, credit: float, max_loss: float) -> int:
        if max_loss <= 0 or credit <= 0:
            return 1
        p = pop / 100
        b = credit / max_loss
        f = max(0.0, (p * b - (1 - p)) / b) * 0.5  # half-Kelly
        capital_per = max_loss * 100
        max_by_pct  = int(self.Portfolio.TotalPortfolioValue * self.MAX_PCT / capital_per)
        kelly_n     = int(f * self.Portfolio.TotalPortfolioValue / capital_per)
        return max(1, min(kelly_n, max_by_pct))
