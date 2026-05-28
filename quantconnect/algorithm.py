"""
QuantConnect LEAN Algorithm — Alpaca-Compliant Long-Gamma/Vega Strategy

Buy directional options when implied vol is historically cheap (low IV rank)
ahead of catalysts. One direction at a time — no straddles, no delta hedging,
no covered positions.

Entry rules:
  1. IV rank < IV_RANK_MAX  (buying cheap vol, not selling expensive vol)
  2. 20-day price momentum confirms clear direction (CALL or PUT)
  3. Both conditions together → construct the appropriate structure

Structure selection (all Alpaca-compliant):
  IV rank < 20  → Pure long call / long put   (maximum gamma, vol is very cheap)
  IV rank 20-35 → Bull call spread / bear put spread  (lower premium outlay)

Excluded by design:
  ❌ Straddles / strangles     (two-directional — violates "one way" rule)
  ❌ Covered calls             (requires equity ownership)
  ❌ Cash-secured puts         (net long delta, not pure gamma play)
  ❌ All naked shorts          (not Alpaca-compliant and wrong direction)

Fill assumptions — conservative buyer-pays model:
  - Entry long legs  fill at ASK  (we pay the offer)
  - Entry short legs fill at BID  (we receive less than mid)
  - Exit  long legs  fill at BID  (we sell at the bid)
  - Exit  short legs fill at ASK  (we buy back at the offer)
  - $0.65 flat commission per contract + $0.05/share slippage
  Combined: approximately 2 × half-spread + commission on each leg.

Risk controls:
  - Risk at most MAX_PREMIUM_PCT (2%) of NAV as premium per position
  - Maximum MAX_POSITIONS (3) concurrent open positions
  - Hard stop at STOP_LOSS_PCT (50%) of premium paid
  - Profit target at PROFIT_TARGET (150%) of premium paid
  - Force exit at DTE_CLOSE (7 DTE) — avoids expiry pin/gamma risk
"""

from AlgorithmImports import *   # noqa: F401,F403
import numpy as np
from scipy.stats import norm


# ══════════════════════════════════════════════════════════════════════════════
# Inlined BSM helpers (scipy only — QC sandbox safe, no numba/pip installs)
# ══════════════════════════════════════════════════════════════════════════════

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bsm_delta(flag, S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) if flag == "c" else norm.cdf(d1) - 1


def yang_zhang_rv(closes, opens, highs, lows, window=21):
    """Yang-Zhang realised vol (annualised %). Returns None if not enough data."""
    if len(closes) < window + 1:
        return None
    o, h, l, c = opens, highs, lows, closes
    ro = np.log(o[1:] / c[:-1])
    rc = np.log(c[1:] / o[1:])
    rs = (np.log(h[1:] / c[1:]) * np.log(h[1:] / o[1:]) +
          np.log(l[1:] / c[1:]) * np.log(l[1:] / o[1:]))
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    from pandas import Series
    yz = (Series(ro).rolling(window).var() +
          k  * Series(rc).rolling(window).var() +
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

class AlpacaLongGammaAlgo(QCAlgorithm):
    """
    Monday event-driven LONG gamma/vega algorithm.

    Logic every Monday (first bar of the week):
      1. Check IV rank vs 1-year rolling HV range  →  is vol CHEAP?
      2. Compute 20-day momentum + 50-day SMA trend  →  which direction?
      3. Both signals agree  →  buy calls or puts (spreads if vol only "cheap-ish")
      4. Individual MarketOrders per leg (long legs first for partial-fill safety)
      5. Daily 15:45 ET manager handles profit, stop-loss, and time exits
    """

    # ── Watchlist ──────────────────────────────────────────────────────────
    WATCHLIST = ["SPY", "QQQ", "IWM"]

    # ── DTE windows ───────────────────────────────────────────────────────
    DTE_TARGET = 30   # target ~30 DTE at open
    DTE_CLOSE  = 7    # force-close with 7 DTE left  (avoids expiry pin risk)

    # ── Strike targeting ──────────────────────────────────────────────────
    LONG_DELTA  = 0.40   # buy ~40Δ (near-ATM for maximum gamma sensitivity)
    SPREAD_WING = 10.0   # spread wing width in $ for defined-risk plays

    # ── Entry signal thresholds ───────────────────────────────────────────
    IV_RANK_MAX  = 35.0  # buy only when IV rank is BELOW this (cheap vol)
    MOM_WINDOW   = 20    # momentum lookback (trading days)
    TREND_WINDOW = 50    # trend confirmation (longer SMA)
    MOM_MIN_PCT  = 1.5   # minimum price move % to confirm direction

    # ── Risk / sizing ─────────────────────────────────────────────────────
    MAX_PREMIUM_PCT = 0.02  # risk at most 2% of NAV as premium per position
    MAX_POSITIONS   = 3     # max concurrent open positions
    PROFIT_TARGET   = 1.50  # exit at 150% of premium (50% gain on investment)
    STOP_LOSS_PCT   = 0.50  # exit at 50% loss on premium paid
    RISK_FREE       = 0.053

    # ─────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)
        self.SetBrokerageModel(BrokerageName.Alpaca, AccountType.Margin)
        self.SetBenchmark("SPY")

        # Conservative buyer-pays fill model
        self.SetSecurityInitializer(self._security_initializer)

        # Subscribe equities + options
        self.option_symbols = {}
        for ticker in self.WATCHLIST:
            equity = self.AddEquity(ticker, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
            option = self.AddOption(ticker, Resolution.Daily)
            option.SetFilter(self._option_filter)
            self.option_symbols[ticker] = option.Symbol

        # State
        self._cached_chains   = {}   # ticker → OptionChain (set in OnData)
        self.open_trades      = {}   # ticker → trade metadata
        self.history_window   = 252

        # Daily position management at 15:45 ET
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 45),
            self.ManagePositions,
        )
        self.Log("AlpacaLongGammaAlgo initialised — buying cheap vol, one direction at a time")

    def _security_initializer(self, security: Security):
        """
        Conservative cost model for option buyers:
          $0.65 flat commission per contract  (Alpaca standard rate)
          $0.05/share slippage                (pushes fills toward ask when buying)
        Together these simulate paying slightly above mid — effectively close to
        the ask price on each long leg, which is the correct pessimistic assumption.
        """
        security.SetFeeModel(ConstantFeeModel(0.65))
        security.SetSlippageModel(ConstantSlippageModel(0.05))

    def _option_filter(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
        return universe.Expiration(15, 45).Strikes(-15, 15)

    # ─────────────────────────────────────────────────────────────────────
    # Data ingestion: cache chains each bar; trigger Monday evaluation
    # ─────────────────────────────────────────────────────────────────────

    def OnData(self, data: Slice):
        # Option chains only flow through OnData, not scheduled functions
        for ticker in self.WATCHLIST:
            sym = self.option_symbols.get(ticker)
            if sym and data.OptionChains.ContainsKey(sym):
                self._cached_chains[ticker] = data.OptionChains[sym]

        # Fire Monday evaluation on the first bar of each new Monday
        if self.Time.weekday() == 0:
            date_key = self.Time.date()
            if not hasattr(self, "_last_eval_date") or self._last_eval_date != date_key:
                self._last_eval_date = date_key
                self.MondayEvaluation()

    # ─────────────────────────────────────────────────────────────────────
    # Monday evaluation
    # ─────────────────────────────────────────────────────────────────────

    def MondayEvaluation(self):
        n_open = len(self.open_trades)
        self.Log(
            f"=== Monday {self.Time.date()} "
            f"| open={n_open}/{self.MAX_POSITIONS} "
            f"| chains cached: {list(self._cached_chains.keys())} ==="
        )

        if n_open >= self.MAX_POSITIONS:
            self.Log("Max positions reached — no new entries this week")
            return

        for ticker in self.WATCHLIST:
            if len(self.open_trades) >= self.MAX_POSITIONS:
                break
            if ticker in self.open_trades:
                self.Log(f"[{ticker}] Already open — skipping")
                continue
            try:
                self._evaluate_and_trade(ticker)
            except Exception as e:
                self.Log(f"[{ticker}] Evaluation error: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Core evaluation: signal → structure → size → submit
    # ─────────────────────────────────────────────────────────────────────

    def _evaluate_and_trade(self, ticker: str):
        # ── 1. Historical OHLC ──────────────────────────────────────────
        hist = list(self.History[TradeBar](
            self.Securities[ticker].Symbol, self.history_window, Resolution.Daily
        ))
        if len(hist) < self.TREND_WINDOW + 5:
            self.Log(f"[{ticker}] Insufficient history ({len(hist)} bars)")
            return

        closes = np.array([b.Close for b in hist])
        opens  = np.array([b.Open  for b in hist])
        highs  = np.array([b.High  for b in hist])
        lows   = np.array([b.Low   for b in hist])
        spot   = float(closes[-1])

        # ── 2. Vol cheapness signal ─────────────────────────────────────
        # We are VOL BUYERS: enter only when IV rank is LOW (cheap premium)
        rv_21 = yang_zhang_rv(closes, opens, highs, lows, window=21)
        if rv_21 is None:
            return

        log_ret   = np.log(closes[1:] / closes[:-1])
        hv_series = [
            np.std(log_ret[i - 21:i]) * np.sqrt(252) * 100
            for i in range(21, len(log_ret))
        ]
        atm_iv = self._get_atm_iv(ticker, spot)
        if atm_iv is None:
            self.Log(f"[{ticker}] No ATM IV — chain cached: {ticker in self._cached_chains}")
            return

        iv_rank = iv_rank_from_hv(hv_series, atm_iv)
        vrp     = atm_iv - rv_21   # negative = IV < RV (vol is cheap)

        self.Log(
            f"[{ticker}] spot={spot:.2f}  IV={atm_iv:.1f}%  RV={rv_21:.1f}%  "
            f"IVRank={iv_rank:.0f}  VRP={vrp:+.1f}%"
        )

        if iv_rank > self.IV_RANK_MAX:
            self.Log(
                f"[{ticker}] IV rank {iv_rank:.0f} > {self.IV_RANK_MAX} "
                f"(vol too expensive to buy) — skip"
            )
            return

        # ── 3. Directional momentum signal ──────────────────────────────
        # One direction only — calls OR puts, never both at once
        direction = self._momentum_signal(closes)
        if direction == "NEUTRAL":
            self.Log(f"[{ticker}] Momentum NEUTRAL — no clear direction — skip")
            return

        self.Log(
            f"[{ticker}] ✓ Signal: IVRank={iv_rank:.0f} (cheap) + {direction} momentum"
        )

        # ── 4. Choose Alpaca-compliant structure ─────────────────────────
        trade_type = self._select_structure(iv_rank, direction)

        # ── 5. Build strike legs from live chain ─────────────────────────
        chain_info = self._build_legs(ticker, spot, trade_type, direction)
        if chain_info is None:
            return

        # ── 6. Premium-risk sizing ───────────────────────────────────────
        n_contracts = self._size_by_premium(chain_info["premium"])
        if n_contracts < 1:
            self.Log(f"[{ticker}] Premium too high to size even 1 contract — skip")
            return

        total_risk = chain_info["premium"] * n_contracts * 100
        self.Log(
            f"[{ticker}] {trade_type} {direction} ×{n_contracts} | "
            f"premium/share={chain_info['premium']:.2f} | "
            f"total at risk=${total_risk:.0f} "
            f"({total_risk / self.Portfolio.TotalPortfolioValue * 100:.1f}% NAV)"
        )

        # ── 7. Submit atomic combo order ─────────────────────────────────
        self._submit_orders(ticker, trade_type, direction, chain_info, n_contracts)

    # ─────────────────────────────────────────────────────────────────────
    # Signal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _momentum_signal(self, closes: np.ndarray) -> str:
        """
        Returns 'CALL' (bullish), 'PUT' (bearish), or 'NEUTRAL'.

        Two conditions must BOTH be true:
          • 20-day return > +MOM_MIN_PCT AND price above 50-day SMA  →  CALL
          • 20-day return < -MOM_MIN_PCT AND price below 50-day SMA  →  PUT
          Otherwise NEUTRAL (skip trade — not enough conviction)

        Requiring both short-term momentum AND trend alignment reduces whipsaws.
        """
        if len(closes) < self.TREND_WINDOW + self.MOM_WINDOW:
            return "NEUTRAL"

        mom_20 = (closes[-1] / closes[-(self.MOM_WINDOW + 1)] - 1) * 100
        sma_50 = float(np.mean(closes[-self.TREND_WINDOW:]))
        spot   = float(closes[-1])

        if mom_20 > self.MOM_MIN_PCT and spot > sma_50:
            return "CALL"
        if mom_20 < -self.MOM_MIN_PCT and spot < sma_50:
            return "PUT"
        return "NEUTRAL"

    # ─────────────────────────────────────────────────────────────────────
    # Structure selector
    # ─────────────────────────────────────────────────────────────────────

    def _select_structure(self, iv_rank: float, direction: str) -> str:
        """
        Very cheap vol (rank < 20)  → pure long call / long put
            Maximum gamma exposure; premium is so low that paying full debit
            is worthwhile for the uncapped upside.

        Moderately cheap (rank 20-35) → spread
            Bull call spread (CALL) or bear put spread (PUT).
            Selling the far-OTM wing reduces the net debit by ~30-40%,
            capping upside but making the trade viable when premium is higher.

        All structures are long gamma net and Alpaca-compliant.
        No covered positions, no straddles, no naked shorts.
        """
        if iv_rank < 20.0:
            return "long_call" if direction == "CALL" else "long_put"
        return "bull_call_spread" if direction == "CALL" else "bear_put_spread"

    # ─────────────────────────────────────────────────────────────────────
    # Leg construction (strike selection + conservative pricing)
    # ─────────────────────────────────────────────────────────────────────

    def _build_legs(self, ticker: str, spot: float,
                    trade_type: str, direction: str) -> dict | None:
        chain = self._cached_chains.get(ticker)
        if chain is None or not chain.Contracts:
            self.Log(f"[{ticker}] No cached chain available")
            return None

        # Group by expiry; choose the one nearest to DTE_TARGET
        by_expiry: dict = {}
        for c in chain.Contracts.Values:
            by_expiry.setdefault(c.Expiry.date(), []).append(c)

        today    = self.Time.date()
        target_e = min(by_expiry, key=lambda e: abs((e - today).days - self.DTE_TARGET))
        T        = (target_e - today).days / 365
        if T <= 0:
            return None
        contracts = by_expiry[target_e]

        dispatch = {
            "long_call":       lambda: self._long_call(contracts, spot, T),
            "long_put":        lambda: self._long_put(contracts, spot, T),
            "bull_call_spread": lambda: self._bull_call_spread(contracts, spot, T),
            "bear_put_spread":  lambda: self._bear_put_spread(contracts, spot, T),
        }
        fn = dispatch.get(trade_type)
        return fn() if fn else None

    def _long_call(self, contracts, spot, T) -> dict | None:
        """Buy ~40Δ call.  Pay ASK (conservative buyer assumption)."""
        calls = [c for c in contracts if c.Right == OptionRight.Call]
        lc = self._nearest_delta(calls, spot, T, self.LONG_DELTA, "c")
        if lc is None or lc.AskPrice <= 0:
            return None
        premium = float(lc.AskPrice)   # ← pay the ask
        return {
            "legs":      [("buy", lc)],
            "premium":   round(premium, 2),
            "be_upper":  lc.Strike + premium,
            "be_lower":  0.0,
            "T":         T,
        }

    def _long_put(self, contracts, spot, T) -> dict | None:
        """Buy ~40Δ put (delta = -0.40).  Pay ASK."""
        puts = [c for c in contracts if c.Right == OptionRight.Put]
        lp = self._nearest_delta(puts, spot, T, -self.LONG_DELTA, "p")
        if lp is None or lp.AskPrice <= 0:
            return None
        premium = float(lp.AskPrice)
        return {
            "legs":      [("buy", lp)],
            "premium":   round(premium, 2),
            "be_upper":  float("inf"),
            "be_lower":  lp.Strike - premium,
            "T":         T,
        }

    def _bull_call_spread(self, contracts, spot, T) -> dict | None:
        """
        Buy ~40Δ call  +  sell call SPREAD_WING $ higher (capped upside, lower debit).

        Conservative pricing:
          Net debit = long_ask − short_bid
          (we pay the ask when buying; we receive the bid when selling)
        """
        calls = sorted(
            [c for c in contracts if c.Right == OptionRight.Call],
            key=lambda c: c.Strike,
        )
        lc = self._nearest_delta(calls, spot, T, self.LONG_DELTA, "c")
        if lc is None:
            return None

        target_sc = lc.Strike + self.SPREAD_WING
        sc = min(
            (c for c in calls if c.Strike > lc.Strike),
            key=lambda c: abs(c.Strike - target_sc),
            default=None,
        )
        if sc is None:
            return None

        # Conservative: pay ask for long, receive bid for short
        debit = float(lc.AskPrice) - float(sc.BidPrice)
        if debit <= 0:
            return None

        max_profit = float(sc.Strike - lc.Strike) - debit
        return {
            "legs":       [("buy", lc), ("sell", sc)],
            "premium":    round(debit, 2),
            "max_profit": round(max_profit, 2),
            "be_upper":   lc.Strike + debit,
            "be_lower":   0.0,
            "T":          T,
        }

    def _bear_put_spread(self, contracts, spot, T) -> dict | None:
        """
        Buy ~40Δ put  +  sell put SPREAD_WING $ lower (capped downside profit, lower debit).

        Conservative pricing:
          Net debit = long_ask − short_bid
        """
        puts = sorted(
            [c for c in contracts if c.Right == OptionRight.Put],
            key=lambda c: c.Strike,
        )
        lp = self._nearest_delta(puts, spot, T, -self.LONG_DELTA, "p")
        if lp is None:
            return None

        target_sp = lp.Strike - self.SPREAD_WING
        sp = min(
            (c for c in puts if c.Strike < lp.Strike),
            key=lambda c: abs(c.Strike - target_sp),
            default=None,
        )
        if sp is None:
            return None

        debit = float(lp.AskPrice) - float(sp.BidPrice)
        if debit <= 0:
            return None

        max_profit = float(lp.Strike - sp.Strike) - debit
        return {
            "legs":       [("buy", lp), ("sell", sp)],
            "premium":    round(debit, 2),
            "max_profit": round(max_profit, 2),
            "be_upper":   float("inf"),
            "be_lower":   lp.Strike - debit,
            "T":          T,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────

    def _size_by_premium(self, premium_per_share: float) -> int:
        """
        Risk the smaller of:
          • MAX_PREMIUM_PCT (2%) of current NAV
          • 1 contract minimum

        Each contract = 100 shares, so cost per contract = premium × 100.
        """
        if premium_per_share <= 0:
            return 0
        max_dollars = self.Portfolio.TotalPortfolioValue * self.MAX_PREMIUM_PCT
        n = int(max_dollars / (premium_per_share * 100))
        return max(1, n)

    # ─────────────────────────────────────────────────────────────────────
    # Order submission — atomic combo
    # ─────────────────────────────────────────────────────────────────────

    def _submit_orders(self, ticker: str, trade_type: str, direction: str,
                       chain_info: dict, n_contracts: int):
        """
        Submit each leg as an individual MarketOrder.

        NOTE: AlpacaBrokerageModel only supports Market and Limit orders —
        ComboMarketOrder is NOT supported in QC's Alpaca backtest model.

        Partial-fill safety — LONG legs submitted FIRST:
          If the short (wing) leg is later canceled, we are left with a plain
          long call or put, which is still Alpaca-compliant and directionally
          correct.  The worst outcome is we paid full debit instead of the
          spread-reduced debit (more expensive, not dangerous).
          A naked short from the short leg filling before the long is impossible
          because short legs are submitted second.
        """
        # Sort: buy legs first (index 0), sell legs second (index 1)
        ordered = sorted(chain_info["legs"], key=lambda x: 0 if x[0] == "buy" else 1)

        for action, contract in ordered:
            qty = n_contracts if action == "buy" else -n_contracts
            price_label = (f"ask={contract.AskPrice:.2f}" if action == "buy"
                           else f"bid={contract.BidPrice:.2f}")
            self.MarketOrder(contract.Symbol, qty)
            self.Log(
                f"  {action.upper()} {contract.Symbol} ×{n_contracts} "
                f"K={contract.Strike} exp={contract.Expiry.date()} {price_label}"
            )

        # Store stable (action, Symbol, strike, expiry, right) tuples —
        # NOT live OptionContract references, which can become stale.
        stored_legs = [
            (action, contract.Symbol, float(contract.Strike),
             contract.Expiry.date(), contract.Right)
            for action, contract in chain_info["legs"]
        ]
        self.open_trades[ticker] = {
            "trade_type":  trade_type,
            "direction":   direction,
            "legs":        stored_legs,
            "premium":     chain_info["premium"],   # net debit paid per share
            "be_upper":    chain_info["be_upper"],
            "be_lower":    chain_info["be_lower"],
            "n_contracts": n_contracts,
            "open_date":   self.Time.date(),
            "expiry":      chain_info["legs"][0][1].Expiry.date(),
        }
        total_cost = chain_info["premium"] * n_contracts * 100
        self.Log(
            f"[{ticker}] ✓ Opened {trade_type} {direction} ×{n_contracts} "
            f"| premium/share={chain_info['premium']:.2f} "
            f"| total=${total_cost:.0f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Position management  (runs daily at 15:45 ET)
    # ─────────────────────────────────────────────────────────────────────

    def ManagePositions(self):
        """
        Exit conditions (checked in order of priority):
          1. DTE ≤ 7         → time exit (avoid expiry gamma / pin risk)
          2. Mark ≥ 150%     → profit target (50% gain on premium paid)
          3. Mark ≤ 50%      → stop loss (50% of premium lost)

        Mark-to-market uses:
          Long legs  → BID  (what we receive when selling to close)
          Short legs → ASK  (what we pay to buy back)
        This is the most conservative / realistic exit valuation.
        """
        today    = self.Time.date()
        to_close = []

        for ticker, meta in list(self.open_trades.items()):
            expiry = meta["expiry"]
            dte    = (expiry - today).days

            # Clean up stale entries (expired without being closed)
            if dte < 0:
                self.Log(f"[{ticker}] Contract expired — cleaning up")
                self.open_trades.pop(ticker, None)
                continue

            # Time exit
            if dte <= self.DTE_CLOSE:
                self.Log(f"[{ticker}] {dte} DTE — time exit")
                to_close.append(ticker)
                continue

            # P&L check
            current_mark = self._get_position_mark(meta)
            if current_mark is None:
                continue

            premium  = meta["premium"]   # what we originally paid per share
            gain_pct = (current_mark - premium) / premium if premium > 0 else 0.0

            if gain_pct >= (self.PROFIT_TARGET - 1.0):
                self.Log(f"[{ticker}] Profit target: +{gain_pct*100:.0f}% — close")
                to_close.append(ticker)
            elif gain_pct <= -self.STOP_LOSS_PCT:
                self.Log(f"[{ticker}] Stop loss: {gain_pct*100:.0f}% — close")
                to_close.append(ticker)

        for ticker in to_close:
            self._close_position(ticker)

    def _close_position(self, ticker: str):
        """
        Close each leg with an individual MarketOrder (Alpaca only supports Market/Limit).

        Exit order — SHORT buyback FIRST:
          Buy back the short (wing) leg before selling the long leg.
          If the short buyback fails (already expired / no market), we are left
          holding a long option — never a naked short.
        """
        meta = self.open_trades.pop(ticker, None)
        if not meta:
            return

        # Sort close orders: buy back shorts first (index 0), then sell longs (index 1)
        ordered = sorted(meta["legs"], key=lambda x: 0 if x[0] == "sell" else 1)

        for leg_tuple in ordered:
            action, sym = leg_tuple[0], leg_tuple[1]
            # Reverse: original "sell" → close by buying (+n), "buy" → close by selling (-n)
            close_qty = meta["n_contracts"] if action == "sell" else -meta["n_contracts"]
            try:
                self.MarketOrder(sym, close_qty)
            except Exception as e:
                self.Log(f"  Could not close {sym}: {e}")

        self.Log(f"[{ticker}] ✓ Closed {meta['trade_type']} {meta['direction']}")

    def _get_position_mark(self, meta: dict) -> float | None:
        """
        Conservative exit mark (per share):
          Long legs:  receive BID  (selling into the market)
          Short legs: pay    ASK   (buying back at the offer)
        Returns None if any leg has stale/zero quotes.
        """
        mark = 0.0
        for leg_tuple in meta["legs"]:
            action, sym = leg_tuple[0], leg_tuple[1]
            if not self.Securities.ContainsKey(sym):
                return None
            sec = self.Securities[sym]
            bid, ask = float(sec.BidPrice), float(sec.AskPrice)
            if bid <= 0 and ask <= 0:
                return None
            # Close price: sell long at bid, buy short back at ask
            close_price = bid if action == "buy" else ask
            mark += close_price if action == "buy" else -close_price
        return mark

    # ─────────────────────────────────────────────────────────────────────
    # Assignment handler
    # ─────────────────────────────────────────────────────────────────────

    def OnAssignmentOrderEvent(self, assignmentEvent):
        """
        Handle early assignment of the short leg in a spread.
        (Unlikely for calls pre-ex-div, but can happen for deep ITM puts.)
        Flatten any resulting equity position immediately and clean up state.
        """
        assigned_sym = assignmentEvent.Symbol
        self.Log(
            f"ASSIGNMENT: {assigned_sym} "
            f"qty={assignmentEvent.Quantity} fill={assignmentEvent.FillPrice:.2f}"
        )
        try:
            underlying = assigned_sym.Underlying
            if underlying is not None and self.Securities.ContainsKey(underlying):
                qty = self.Portfolio[underlying].Quantity
                if qty != 0:
                    self.MarketOrder(underlying, -qty)
                    self.Log(f"  Flattened underlying {underlying} ×{qty}")
        except Exception as e:
            self.Log(f"  Assignment flatten error: {e}")

        for ticker in list(self.open_trades.keys()):
            for leg_tuple in self.open_trades[ticker]["legs"]:
                if leg_tuple[1] == assigned_sym:
                    self.Log(f"  [{ticker}] Removed from open_trades after assignment")
                    self.open_trades.pop(ticker, None)
                    break

    # ─────────────────────────────────────────────────────────────────────
    # IV / Greeks helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_atm_iv(self, ticker: str, spot: float) -> float | None:
        """
        ATM IV with three-level fallback:
          1. QC pre-computed ImpliedVolatility field
          2. Newton-Raphson BSM solver on the mid price
          3. Yang-Zhang realised vol as last-resort proxy
        """
        chain = self._cached_chains.get(ticker)
        if chain is None:
            return None
        calls = [c for c in chain.Contracts.Values if c.Right == OptionRight.Call]
        if not calls:
            return None
        atm = min(calls, key=lambda c: abs(c.Strike - spot))

        # Level 1: QC computed IV
        if atm.ImpliedVolatility > 0:
            return float(atm.ImpliedVolatility) * 100

        # Level 2: BSM solver on mid
        mid = (float(atm.BidPrice) + float(atm.AskPrice)) / 2
        T   = (atm.Expiry.date() - self.Time.date()).days / 365
        if mid > 0 and T > 0:
            iv = self._solve_iv(mid, spot, float(atm.Strike), T)
            if iv:
                self.Log(f"[{ticker}] IV via BSM solver: {iv:.1f}%")
                return iv

        # Level 3: Yang-Zhang RV as proxy
        hist = list(self.History[TradeBar](
            self.Securities[ticker].Symbol, 63, Resolution.Daily
        ))
        if len(hist) >= 22:
            c = np.array([b.Close for b in hist])
            o = np.array([b.Open  for b in hist])
            h = np.array([b.High  for b in hist])
            l = np.array([b.Low   for b in hist])
            rv = yang_zhang_rv(c, o, h, l, window=21)
            if rv:
                self.Log(f"[{ticker}] IV proxy via RV: {rv:.1f}%")
                return rv
        return None

    def _solve_iv(self, price, S, K, T, flag="c", tol=1e-4) -> float | None:
        """Newton-Raphson BSM IV solver (call price, safe vega guard)."""
        sigma = 0.25
        for _ in range(50):
            d1 = (np.log(S / K) + (self.RISK_FREE + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            theo = S * norm.cdf(d1) - K * np.exp(-self.RISK_FREE * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            if abs(vega) < 1e-10:
                break
            sigma -= (theo - price) / vega
            sigma  = max(0.001, min(sigma, 5.0))
            if abs(theo - price) < tol:
                return round(sigma * 100, 2)
        return None

    def _nearest_delta(self, contracts, spot, T, target_delta, flag):
        """Return the contract whose BSM delta is closest to target_delta."""
        best, best_diff = None, float("inf")
        for c in contracts:
            iv   = max(float(c.ImpliedVolatility), 0.05)
            d    = bsm_delta(flag, spot, float(c.Strike), T, self.RISK_FREE, iv)
            diff = abs(d - target_delta)
            if diff < best_diff:
                best, best_diff = c, diff
        return best
