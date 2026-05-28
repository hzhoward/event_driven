# Long-gamma/vega event-driven strategy — Alpaca-compliant
# Buy cheap vol (low IV rank) + directional momentum = long call or put (spread if vol moderate)
# One direction at a time. No straddles. No covered delta. No naked shorts.
# Fill model: long legs pay ASK, short legs receive BID. $0.65 commission + $0.05 slippage.
from AlgorithmImports import *
import numpy as np
from scipy.stats import norm

# ── BSM helpers (inlined — no pip install needed in QC sandbox) ──────────────

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bsm_delta(flag, S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) if flag == "c" else norm.cdf(d1) - 1

def yang_zhang_rv(closes, opens, highs, lows, window=21):
    if len(closes) < window + 1:
        return None
    ro = np.log(opens[1:] / closes[:-1])
    rc = np.log(closes[1:] / opens[1:])
    rs = (np.log(highs[1:] / closes[1:]) * np.log(highs[1:] / opens[1:]) +
          np.log(lows[1:]  / closes[1:]) * np.log(lows[1:]  / opens[1:]))
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

# ── Algorithm ─────────────────────────────────────────────────────────────────

class AlpacaLongGammaAlgo(QCAlgorithm):

    # Watchlist — ETFs + high-IV single stocks
    WATCHLIST = [
        "SPY",  "QQQ",  "IWM",
        "NVDA", "AMD",  "AVGO", "MSFT", "META",
        "PLTR", "ANET", "VRT",  "GEV",  "CRDO",
        "TSM",  "ORCL", "PANW", "NOW",  "ASML",
        "AMAT", "NFLX", "NEE",  "EQIX",
        "DLR",  "AMT",  "TSLA", "WDC",  "INTC",
    ]

    # DTE
    DTE_TARGET = 30
    DTE_CLOSE  = 7

    # Strike targeting
    LONG_DELTA  = 0.40   # near-ATM for maximum gamma
    SPREAD_WING = 0.03   # 3 % of spot price (works across all price levels)

    # Entry thresholds
    IV_RANK_MAX  = 40.0  # enter only when vol is cheap (rank below this)
    MOM_WINDOW   = 20
    TREND_WINDOW = 50
    MOM_MIN_PCT  = 1.5   # min 1.5 % move to confirm direction

    # Risk / sizing
    MAX_PREMIUM_PCT = 0.02   # risk at most 2 % of NAV per position
    MAX_POSITIONS   = 5      # up to 5 concurrent open positions
    PROFIT_TARGET   = 1.50   # close at 150 % of premium (50 % gain)
    STOP_LOSS_PCT   = 0.50   # close at 50 % loss
    RISK_FREE       = 0.053

    # ── Init ─────────────────────────────────────────────────────────────────

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)
        # Use IB brokerage model for backtesting — QC's Alpaca model rejects
        # ComboMarket orders that LEAN auto-generates when multiple option legs
        # on the same underlying are submitted in the same bar.
        # Strategy logic remains Alpaca-compliant (long calls/puts/spreads only).
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetBenchmark("SPY")
        self.SetSecurityInitializer(self._sec_init)

        self.option_symbols = {}
        for ticker in self.WATCHLIST:
            eq = self.AddEquity(ticker, Resolution.Daily)
            eq.SetDataNormalizationMode(DataNormalizationMode.Raw)
            opt = self.AddOption(ticker, Resolution.Daily)
            opt.SetFilter(self._opt_filter)
            self.option_symbols[ticker] = opt.Symbol

        self._chains   = {}   # ticker → OptionChain  (populated in OnData)
        self.open_trades = {}
        self.hist_bars   = 252

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 45),
            self.ManagePositions,
        )
        self.Log("AlpacaLongGammaAlgo ready")

    def _sec_init(self, security):
        security.SetFeeModel(ConstantFeeModel(0.65))
        security.SetSlippageModel(ConstantSlippageModel(0.05))

    def _opt_filter(self, u):
        return u.Expiration(15, 45).Strikes(-20, 20)

    # ── OnData: cache chains, fire Monday evaluation ──────────────────────────

    def OnData(self, data):
        for ticker in self.WATCHLIST:
            sym = self.option_symbols.get(ticker)
            if sym and data.OptionChains.ContainsKey(sym):
                self._chains[ticker] = data.OptionChains[sym]

        if self.Time.weekday() == 0:
            dk = self.Time.date()
            if not hasattr(self, "_last_eval") or self._last_eval != dk:
                self._last_eval = dk
                self.MondayEvaluation()

    # ── Monday evaluation ─────────────────────────────────────────────────────

    def MondayEvaluation(self):
        n = len(self.open_trades)
        self.Log(f"=== Monday {self.Time.date()} open={n}/{self.MAX_POSITIONS} "
                 f"chains={list(self._chains.keys())} ===")
        if n >= self.MAX_POSITIONS:
            return
        for ticker in self.WATCHLIST:
            if len(self.open_trades) >= self.MAX_POSITIONS:
                break
            if ticker in self.open_trades:
                continue
            try:
                self._evaluate(ticker)
            except Exception as e:
                self.Log(f"[{ticker}] error: {e}")

    # ── Core signal + trade logic ─────────────────────────────────────────────

    def _evaluate(self, ticker):
        hist = list(self.History[TradeBar](
            self.Securities[ticker].Symbol, self.hist_bars, Resolution.Daily))
        if len(hist) < self.TREND_WINDOW + 5:
            return

        closes = np.array([b.Close for b in hist])
        opens  = np.array([b.Open  for b in hist])
        highs  = np.array([b.High  for b in hist])
        lows   = np.array([b.Low   for b in hist])
        spot   = float(closes[-1])

        rv21 = yang_zhang_rv(closes, opens, highs, lows, 21)
        if rv21 is None:
            return

        log_ret  = np.log(closes[1:] / closes[:-1])
        hv_series = [np.std(log_ret[i-21:i]) * np.sqrt(252) * 100
                     for i in range(21, len(log_ret))]

        atm_iv = self._get_atm_iv(ticker, spot)
        if atm_iv is None:
            self.Log(f"[{ticker}] no ATM IV")
            return

        iv_rank = iv_rank_from_hv(hv_series, atm_iv)
        vrp     = atm_iv - rv21
        self.Log(f"[{ticker}] spot={spot:.2f} IV={atm_iv:.1f}% RV={rv21:.1f}% "
                 f"IVR={iv_rank:.0f} VRP={vrp:+.1f}%")

        # Vol cheapness filter — we are buyers
        if iv_rank > self.IV_RANK_MAX:
            self.Log(f"[{ticker}] IV rank {iv_rank:.0f} > {self.IV_RANK_MAX} — skip")
            return

        direction = self._momentum(closes)
        if direction == "NEUTRAL":
            self.Log(f"[{ticker}] momentum neutral — skip")
            return

        self.Log(f"[{ticker}] ✓ IVR={iv_rank:.0f} cheap + {direction}")
        trade_type = self._structure(iv_rank, direction)
        info = self._build_legs(ticker, spot, trade_type)
        if info is None:
            return

        n_contracts = self._size(info["premium"])
        if n_contracts < 1:
            return

        total = info["premium"] * n_contracts * 100
        self.Log(f"[{ticker}] {trade_type} {direction} x{n_contracts} "
                 f"prem={info['premium']:.2f} total=${total:.0f}")
        self._submit(ticker, trade_type, direction, info, n_contracts)

    # ── Signals ───────────────────────────────────────────────────────────────

    def _momentum(self, closes):
        if len(closes) < self.TREND_WINDOW + self.MOM_WINDOW:
            return "NEUTRAL"
        mom = (closes[-1] / closes[-(self.MOM_WINDOW+1)] - 1) * 100
        sma = float(np.mean(closes[-self.TREND_WINDOW:]))
        spot = float(closes[-1])
        if mom >  self.MOM_MIN_PCT and spot > sma:
            return "CALL"
        if mom < -self.MOM_MIN_PCT and spot < sma:
            return "PUT"
        return "NEUTRAL"

    def _structure(self, iv_rank, direction):
        # Very cheap vol → pure long (max gamma); moderate → spread (cheaper debit)
        if iv_rank < 20.0:
            return "long_call" if direction == "CALL" else "long_put"
        return "bull_call_spread" if direction == "CALL" else "bear_put_spread"

    # ── Leg construction ──────────────────────────────────────────────────────

    def _build_legs(self, ticker, spot, trade_type):
        chain = self._chains.get(ticker)
        if chain is None or not chain.Contracts:
            return None

        by_exp = {}
        for c in chain.Contracts.Values:
            by_exp.setdefault(c.Expiry.date(), []).append(c)

        today = self.Time.date()
        exp   = min(by_exp, key=lambda e: abs((e - today).days - self.DTE_TARGET))
        T     = (exp - today).days / 365
        if T <= 0:
            return None
        cs = by_exp[exp]

        if trade_type == "long_call":
            return self._long_call(cs, spot, T)
        if trade_type == "long_put":
            return self._long_put(cs, spot, T)
        if trade_type == "bull_call_spread":
            return self._bcs(cs, spot, T)
        if trade_type == "bear_put_spread":
            return self._bps(cs, spot, T)
        return None

    def _long_call(self, cs, spot, T):
        calls = [c for c in cs if c.Right == OptionRight.Call]
        lc = self._nearest_delta(calls, spot, T, self.LONG_DELTA, "c")
        if lc is None or lc.AskPrice <= 0:
            return None
        p = float(lc.AskPrice)
        return {"legs": [("buy", lc)], "premium": round(p, 2),
                "be_upper": lc.Strike + p, "be_lower": 0.0}

    def _long_put(self, cs, spot, T):
        puts = [c for c in cs if c.Right == OptionRight.Put]
        lp = self._nearest_delta(puts, spot, T, -self.LONG_DELTA, "p")
        if lp is None or lp.AskPrice <= 0:
            return None
        p = float(lp.AskPrice)
        return {"legs": [("buy", lp)], "premium": round(p, 2),
                "be_upper": float("inf"), "be_lower": lp.Strike - p}

    def _bcs(self, cs, spot, T):
        # Bull call spread: buy ~40Δ call, sell call SPREAD_WING% higher
        calls = sorted([c for c in cs if c.Right == OptionRight.Call],
                       key=lambda c: c.Strike)
        lc = self._nearest_delta(calls, spot, T, self.LONG_DELTA, "c")
        if lc is None:
            return None
        wing   = spot * self.SPREAD_WING
        target = lc.Strike + wing
        sc = min((c for c in calls if c.Strike > lc.Strike),
                 key=lambda c: abs(c.Strike - target), default=None)
        if sc is None:
            return None
        debit = float(lc.AskPrice) - float(sc.BidPrice)
        if debit <= 0:
            return None
        return {"legs": [("buy", lc), ("sell", sc)], "premium": round(debit, 2),
                "max_profit": round(float(sc.Strike - lc.Strike) - debit, 2),
                "be_upper": lc.Strike + debit, "be_lower": 0.0}

    def _bps(self, cs, spot, T):
        # Bear put spread: buy ~40Δ put, sell put SPREAD_WING% lower
        puts = sorted([c for c in cs if c.Right == OptionRight.Put],
                      key=lambda c: c.Strike)
        lp = self._nearest_delta(puts, spot, T, -self.LONG_DELTA, "p")
        if lp is None:
            return None
        wing   = spot * self.SPREAD_WING
        target = lp.Strike - wing
        sp = min((c for c in puts if c.Strike < lp.Strike),
                 key=lambda c: abs(c.Strike - target), default=None)
        if sp is None:
            return None
        debit = float(lp.AskPrice) - float(sp.BidPrice)
        if debit <= 0:
            return None
        return {"legs": [("buy", lp), ("sell", sp)], "premium": round(debit, 2),
                "max_profit": round(float(lp.Strike - sp.Strike) - debit, 2),
                "be_upper": float("inf"), "be_lower": lp.Strike - debit}

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _size(self, premium):
        if premium <= 0:
            return 0
        max_risk = self.Portfolio.TotalPortfolioValue * self.MAX_PREMIUM_PCT
        return max(1, int(max_risk / (premium * 100)))

    # ── Order submission ──────────────────────────────────────────────────────

    def _submit(self, ticker, trade_type, direction, info, n):
        # AlpacaBrokerageModel supports Market/Limit only — no ComboMarketOrder.
        # Safety: submit LONG legs first so any partial fill leaves a long position,
        # never a naked short.
        ordered = sorted(info["legs"], key=lambda x: 0 if x[0] == "buy" else 1)
        for action, contract in ordered:
            qty = n if action == "buy" else -n
            self.MarketOrder(contract.Symbol, qty)
            self.Log(f"  {action.upper()} {contract.Symbol} x{n} K={contract.Strike}")

        stored = [(a, c.Symbol, float(c.Strike), c.Expiry.date(), c.Right)
                  for a, c in info["legs"]]
        self.open_trades[ticker] = {
            "trade_type": trade_type, "direction": direction,
            "legs": stored, "premium": info["premium"],
            "be_upper": info["be_upper"], "be_lower": info["be_lower"],
            "n_contracts": n, "open_date": self.Time.date(),
            "expiry": info["legs"][0][1].Expiry.date(),
        }

    # ── Position management (daily 15:45 ET) ──────────────────────────────────

    def ManagePositions(self):
        today    = self.Time.date()
        to_close = []
        for ticker, meta in list(self.open_trades.items()):
            dte = (meta["expiry"] - today).days
            if dte < 0:
                self.open_trades.pop(ticker, None)
                continue
            if dte <= self.DTE_CLOSE:
                self.Log(f"[{ticker}] {dte} DTE — time exit")
                to_close.append(ticker)
                continue
            mark = self._mark(meta)
            if mark is None:
                continue
            prem  = meta["premium"]
            gain  = (mark - prem) / prem if prem > 0 else 0.0
            if gain >= (self.PROFIT_TARGET - 1.0):
                self.Log(f"[{ticker}] profit target +{gain*100:.0f}%")
                to_close.append(ticker)
            elif gain <= -self.STOP_LOSS_PCT:
                self.Log(f"[{ticker}] stop loss {gain*100:.0f}%")
                to_close.append(ticker)
        for t in to_close:
            self._close(t)

    def _close(self, ticker):
        meta = self.open_trades.pop(ticker, None)
        if not meta:
            return
        # Buy back shorts first, then sell longs — safe partial-fill ordering
        ordered = sorted(meta["legs"], key=lambda x: 0 if x[0] == "sell" else 1)
        for leg in ordered:
            action, sym = leg[0], leg[1]
            qty = meta["n_contracts"] if action == "sell" else -meta["n_contracts"]
            try:
                self.MarketOrder(sym, qty)
            except Exception as e:
                self.Log(f"  close {sym} failed: {e}")
        self.Log(f"[{ticker}] closed {meta['trade_type']} {meta['direction']}")

    def _mark(self, meta):
        # Conservative: long legs at BID, short legs at ASK
        mark = 0.0
        for leg in meta["legs"]:
            action, sym = leg[0], leg[1]
            if not self.Securities.ContainsKey(sym):
                return None
            sec = self.Securities[sym]
            bid, ask = float(sec.BidPrice), float(sec.AskPrice)
            if bid <= 0 and ask <= 0:
                return None
            price = bid if action == "buy" else ask
            mark += price if action == "buy" else -price
        return mark

    # ── Assignment handler ────────────────────────────────────────────────────

    def OnAssignmentOrderEvent(self, ev):
        sym = ev.Symbol
        self.Log(f"ASSIGNMENT {sym} qty={ev.Quantity}")
        try:
            ul = sym.Underlying
            if ul and self.Securities.ContainsKey(ul):
                qty = self.Portfolio[ul].Quantity
                if qty != 0:
                    self.MarketOrder(ul, -qty)
                    self.Log(f"  flattened {ul} x{qty}")
        except Exception as e:
            self.Log(f"  assignment flatten error: {e}")
        for ticker in list(self.open_trades.keys()):
            for leg in self.open_trades[ticker]["legs"]:
                if leg[1] == sym:
                    self.open_trades.pop(ticker, None)
                    break

    # ── IV / Greek helpers ────────────────────────────────────────────────────

    def _get_atm_iv(self, ticker, spot):
        chain = self._chains.get(ticker)
        if chain is None:
            return None
        calls = [c for c in chain.Contracts.Values if c.Right == OptionRight.Call]
        if not calls:
            return None
        atm = min(calls, key=lambda c: abs(c.Strike - spot))
        if atm.ImpliedVolatility > 0:
            return float(atm.ImpliedVolatility) * 100
        mid = (float(atm.BidPrice) + float(atm.AskPrice)) / 2
        T   = (atm.Expiry.date() - self.Time.date()).days / 365
        if mid > 0 and T > 0:
            iv = self._solve_iv(mid, spot, float(atm.Strike), T)
            if iv:
                return iv
        hist = list(self.History[TradeBar](
            self.Securities[ticker].Symbol, 63, Resolution.Daily))
        if len(hist) >= 22:
            rv = yang_zhang_rv(
                np.array([b.Close for b in hist]),
                np.array([b.Open  for b in hist]),
                np.array([b.High  for b in hist]),
                np.array([b.Low   for b in hist]), 21)
            if rv:
                return rv
        return None

    def _solve_iv(self, price, S, K, T, tol=1e-4):
        sigma = 0.25
        for _ in range(50):
            d1 = (np.log(S/K) + (self.RISK_FREE + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            theo = S*norm.cdf(d1) - K*np.exp(-self.RISK_FREE*T)*norm.cdf(d2)
            vega = S*norm.pdf(d1)*np.sqrt(T)
            if abs(vega) < 1e-10:
                break
            sigma -= (theo - price) / vega
            sigma  = max(0.001, min(sigma, 5.0))
            if abs(theo - price) < tol:
                return round(sigma * 100, 2)
        return None

    def _nearest_delta(self, contracts, spot, T, target, flag):
        best, bd = None, float("inf")
        for c in contracts:
            iv = max(float(c.ImpliedVolatility), 0.05)
            d  = bsm_delta(flag, spot, float(c.Strike), T, self.RISK_FREE, iv)
            if abs(d - target) < bd:
                best, bd = c, abs(d - target)
        return best
