"""
LongGammaStrategy — Backtrader implementation of the event-driven long-gamma algo.

Since Backtrader has no native options data support, this module uses
SYNTHETIC options pricing:

  1. Estimate IV = Yang-Zhang RV × IV_PREMIUM (typical equity IV risk premium ~1.15)
  2. Price each option leg using Black-Scholes (bsm_price from options_engine)
  3. Find target-delta strikes via brentq root-finding on the BSM delta function
  4. Simulate bid/ask spread: buyers pay BSM × (1+slip), sellers receive BSM × (1-slip)
  5. Track position value with daily BSM mark-to-market; apply profit/stop/time exits

This replicates the same logic as algorithm_qc.py without needing live option chain
data.  The main limitation is that IV estimation is less precise than observed market
IV — results are directionally correct but conservative.

Reuses from options_engine:
  - options_engine.pricer.bsm_price   (BSM theoretical price)
  - options_engine.pricer.bsm_delta   (for strike-by-delta search)
  - Yang-Zhang RV math (inlined as numpy arrays for BT data feed compatibility)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import backtrader as bt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

log = logging.getLogger(__name__)

# Import BSM functions from the shared options engine
try:
    from options_engine.pricer import bsm_price, bsm_delta
except ImportError:
    # Fallback: inline definitions so the file runs standalone too
    def _d1d2(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return d1, d1 - sigma * np.sqrt(T)

    def bsm_price(flag, S, K, T, r, sigma):
        d1, d2 = _d1d2(S, K, T, r, sigma)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return float(np.where(np.asarray(flag) == "c", call, put))

    def bsm_delta(flag, S, K, T, r, sigma):
        d1, _ = _d1d2(S, K, T, r, sigma)
        return float(np.where(np.asarray(flag) == "c", norm.cdf(d1), norm.cdf(d1) - 1))


# ── Yang-Zhang RV (numpy arrays — BT data feed compatible) ───────────────────

def _yz_rv(closes, opens, highs, lows, window=21) -> Optional[float]:
    """Yang-Zhang realised vol from numpy arrays. Returns annualised % or None."""
    if len(closes) < window + 2:
        return None
    ro = np.log(opens[1:] / closes[:-1])
    rc = np.log(closes[1:] / opens[1:])
    rs = (np.log(highs[1:] / closes[1:]) * np.log(highs[1:] / opens[1:]) +
          np.log(lows[1:]  / closes[1:]) * np.log(lows[1:]  / opens[1:]))
    k   = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = (np.array([
               np.var(ro[i-window:i], ddof=1)
               + k * np.var(rc[i-window:i], ddof=1)
               + (1 - k) * np.mean(rs[i-window:i])
               for i in range(window, len(ro))
           ])[-1])
    return float(np.sqrt(max(var, 0) * 252) * 100)


def _iv_rank(hv_series: list[float], current_iv: float) -> float:
    lo, hi = min(hv_series), max(hv_series)
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (current_iv - lo) / (hi - lo) * 100))


# ── Strategy ──────────────────────────────────────────────────────────────────

class LongGammaStrategy(bt.Strategy):
    """
    Monday event-driven long-gamma strategy.

    Signal:
      - IV rank < iv_rank_max  (buy cheap vol)
      - 20-day momentum + 50-day trend agree on direction
      → construct long call / long put / bull-call-spread / bear-put-spread

    All exits:
      - Profit target : mark ≥ profit_target × premium paid
      - Stop loss     : mark ≤ (1 - stop_loss_pct) × premium paid
      - Time exit     : dte_close days before synthetic expiry
    """

    params = (
        # ── Signal ──────────────────────────────────────────────────────
        ("iv_rank_max",    40.0),  # enter only when IV rank is below this
        ("mom_window",     20),    # momentum lookback (trading days)
        ("trend_window",   50),    # longer SMA for trend confirmation
        ("mom_min_pct",    1.5),   # minimum move % to confirm direction
        # ── Risk / sizing ───────────────────────────────────────────────
        ("max_premium_pct", 0.02), # max 2 % of NAV as premium per trade
        ("max_positions",   5),    # max concurrent positions
        ("profit_target",   1.50), # close at 150 % of premium (50 % gain)
        ("stop_loss_pct",   0.50), # close at 50 % premium loss
        # ── Options ─────────────────────────────────────────────────────
        ("dte_target",     30),    # target DTE at entry
        ("dte_close",      7),     # force-close with ≤7 DTE
        ("long_delta",     0.40),  # target ~40Δ strike (near-ATM, max gamma)
        ("spread_wing_pct", 0.03), # spread wing = 3 % of spot
        # ── IV estimation ───────────────────────────────────────────────
        ("iv_premium",     1.15),  # IV = RV × this (typical equity vol premium)
        ("risk_free",      0.053),
        # ── Fill simulation ─────────────────────────────────────────────
        ("slip",           0.01),  # 1 % of BSM price to simulate bid/ask spread
        ("commission",     0.65),  # $ per contract per leg (matches OptionsCommission)
    )

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        # Map ticker name → BT data object for quick lookup
        self.data_map: dict[str, bt.DataBase] = {d._name: d for d in self.datas}

        # Synthetic option positions
        # ticker → {trade_type, direction, legs, premium, n_contracts, ...}
        self.open_trades: dict[str, dict] = {}

        # Completed trade records (for post-run reporting)
        self.trade_log: list[dict] = []

        self._last_monday: Optional[date] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def next(self):
        today = self.datas[0].datetime.date(0)

        # Monday evaluation — once per week
        if today.weekday() == 0 and today != self._last_monday:
            self._last_monday = today
            self._monday_eval(today)

        # Daily position management
        self._manage_positions(today)

    # ─────────────────────────────────────────────────────────────────────────
    # Monday evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _monday_eval(self, today: date):
        n_open = len(self.open_trades)
        self.log(f"=== Monday {today}  open={n_open}/{self.p.max_positions} ===")
        if n_open >= self.p.max_positions:
            return
        for ticker, data in self.data_map.items():
            if len(self.open_trades) >= self.p.max_positions:
                break
            if ticker in self.open_trades:
                continue
            if len(data) < self.p.trend_window + self.p.mom_window + 5:
                continue
            try:
                self._evaluate(ticker, data, today)
            except Exception as e:
                self.log(f"[{ticker}] evaluation error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Core evaluation: signal → structure → price → submit
    # ─────────────────────────────────────────────────────────────────────────

    # ── Safe OHLCV reader ────────────────────────────────────────────────────
    # data.close.get(size=n) has a ring-buffer zero-fill bug on early bars:
    # it tries to read before the start of the array and returns zeros, making
    # the spot price appear stuck at the first bar's value.
    # Direct indexing (data.close[-i]) is always safe.

    def _read_ohlcv(self, data, max_bars: int = 252):
        """
        Read up to max_bars of OHLCV from a BT data using safe direct indexing.
        Returns four numpy arrays (oldest→newest) or None if insufficient data.
        """
        avail = len(data) - 1          # bars we can look back (excl. current)
        n     = min(avail, max_bars - 1)
        if n < self.p.trend_window + self.p.mom_window:
            return None, None, None, None

        # data.close[-i] = i bars ago;  data.close[0] = current bar
        # Build chronological array: index 0 = oldest, index -1 = current
        indices = list(range(n, -1, -1))   # [n, n-1, ..., 1, 0]
        closes = np.array([float(data.close[-i] if i else data.close[0]) for i in indices])
        opens  = np.array([float(data.open[-i]  if i else data.open[0])  for i in indices])
        highs  = np.array([float(data.high[-i]  if i else data.high[0])  for i in indices])
        lows   = np.array([float(data.low[-i]   if i else data.low[0])   for i in indices])
        return closes, opens, highs, lows

    def _evaluate(self, ticker: str, data: bt.DataBase, today: date):
        # ── 1. OHLCV arrays ─────────────────────────────────────────────
        closes, opens, highs, lows = self._read_ohlcv(data, 252)
        if closes is None:
            return
        spot = float(closes[-1])

        # ── 2. IV estimate ───────────────────────────────────────────────
        # Yang-Zhang RV scaled by iv_premium to approximate ATM IV
        rv21 = _yz_rv(closes, opens, highs, lows, window=21)
        if rv21 is None or rv21 <= 0:
            return
        sigma = (rv21 * self.p.iv_premium) / 100   # decimal (e.g., 0.20 = 20%)

        # ── 3. IV rank ───────────────────────────────────────────────────
        log_ret   = np.log(closes[1:] / closes[:-1])
        hv_series = [
            float(np.std(log_ret[i-21:i], ddof=1)) * np.sqrt(252) * 100
            for i in range(21, len(log_ret))
        ]
        if len(hv_series) < 5:
            return
        iv_pct = rv21 * self.p.iv_premium   # IV as % annualised
        rank   = _iv_rank(hv_series, iv_pct)

        self.log(f"[{ticker}] spot={spot:.2f}  RV={rv21:.1f}%  "
                 f"IV≈{iv_pct:.1f}%  IVRank={rank:.0f}")

        # ── 4. Vol cheapness filter ──────────────────────────────────────
        if rank > self.p.iv_rank_max:
            self.log(f"[{ticker}] IVRank={rank:.0f} > {self.p.iv_rank_max} — skip")
            return

        # ── 5. Directional momentum ──────────────────────────────────────
        direction = self._momentum(closes)
        if direction == "NEUTRAL":
            self.log(f"[{ticker}] momentum neutral — skip")
            return

        self.log(f"[{ticker}] ✓ cheap vol (rank={rank:.0f}) + {direction}")

        # ── 6. Structure selection ───────────────────────────────────────
        trade_type = self._structure(rank, direction)

        # ── 7. Build synthetic legs ──────────────────────────────────────
        T       = self.p.dte_target / 365
        expiry  = today + timedelta(days=self.p.dte_target)
        info    = self._build_legs(spot, sigma, T, trade_type, direction)
        if info is None:
            self.log(f"[{ticker}] could not price legs — skip")
            return

        # ── 8. Position sizing ───────────────────────────────────────────
        n_contracts = self._size(info["premium"])
        if n_contracts < 1:
            return

        # ── 9. Total cost including commission ───────────────────────────
        n_legs      = len(info["legs"])
        premium_tot = info["premium"] * n_contracts * 100
        commission  = n_legs * n_contracts * self.p.commission
        total_debit = premium_tot + commission

        cash = self.broker.getcash()
        if cash < total_debit:
            self.log(f"[{ticker}] insufficient cash ${cash:.0f} < ${total_debit:.0f}")
            return

        # ── 10. Deduct cash (simulate buying options) ────────────────────
        self.broker.add_cash(-total_debit)

        self.open_trades[ticker] = {
            "trade_type":   trade_type,
            "direction":    direction,
            "legs":         info["legs"],   # list of (action, flag, K)
            "entry_sigma":  sigma,
            "entry_spot":   spot,
            "premium":      info["premium"],
            "n_contracts":  n_contracts,
            "total_cost":   total_debit,
            "open_date":    today,
            "expiry":       expiry,
        }
        self.log(
            f"[{ticker}] OPEN {trade_type} {direction} ×{n_contracts} "
            f"prem/share={info['premium']:.2f}  total=${total_debit:.0f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Signals
    # ─────────────────────────────────────────────────────────────────────────

    def _momentum(self, closes: np.ndarray) -> str:
        """Returns 'CALL', 'PUT', or 'NEUTRAL'."""
        min_len = self.p.trend_window + self.p.mom_window
        if len(closes) < min_len:
            return "NEUTRAL"
        mom  = (closes[-1] / closes[-(self.p.mom_window + 1)] - 1) * 100
        sma  = float(np.mean(closes[-self.p.trend_window:]))
        spot = float(closes[-1])
        if mom >  self.p.mom_min_pct and spot > sma:
            return "CALL"
        if mom < -self.p.mom_min_pct and spot < sma:
            return "PUT"
        return "NEUTRAL"

    def _structure(self, iv_rank: float, direction: str) -> str:
        """
        Very cheap vol (rank<20) → pure long for max gamma exposure.
        Moderately cheap    → spread to reduce premium outlay.
        """
        if iv_rank < 20.0:
            return "long_call" if direction == "CALL" else "long_put"
        return "bull_call_spread" if direction == "CALL" else "bear_put_spread"

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic leg construction (BSM pricing + brentq strike search)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_legs(self, spot, sigma, T, trade_type, direction) -> Optional[dict]:
        r = self.p.risk_free
        if trade_type == "long_call":
            K = self._strike_for_delta(spot, T, r, sigma, self.p.long_delta, "c")
            p = self._theo_price("c", spot, K, T, r, sigma, buyer=True)
            return {"legs": [("buy", "c", K)], "premium": round(p, 2)}

        if trade_type == "long_put":
            K = self._strike_for_delta(spot, T, r, sigma, -self.p.long_delta, "p")
            p = self._theo_price("p", spot, K, T, r, sigma, buyer=True)
            return {"legs": [("buy", "p", K)], "premium": round(p, 2)}

        if trade_type == "bull_call_spread":
            K_l = self._strike_for_delta(spot, T, r, sigma, self.p.long_delta, "c")
            K_s = K_l + spot * self.p.spread_wing_pct
            K_s = round(K_s * 2) / 2   # round to nearest $0.50
            p_l = self._theo_price("c", spot, K_l, T, r, sigma, buyer=True)
            p_s = self._theo_price("c", spot, K_s, T, r, sigma, buyer=False)
            debit = p_l - p_s
            if debit <= 0:
                return None
            return {"legs": [("buy", "c", K_l), ("sell", "c", K_s)],
                    "premium": round(debit, 2)}

        if trade_type == "bear_put_spread":
            K_l = self._strike_for_delta(spot, T, r, sigma, -self.p.long_delta, "p")
            K_s = K_l - spot * self.p.spread_wing_pct
            K_s = round(K_s * 2) / 2
            p_l = self._theo_price("p", spot, K_l, T, r, sigma, buyer=True)
            p_s = self._theo_price("p", spot, K_s, T, r, sigma, buyer=False)
            debit = p_l - p_s
            if debit <= 0:
                return None
            return {"legs": [("buy", "p", K_l), ("sell", "p", K_s)],
                    "premium": round(debit, 2)}

        return None

    def _strike_for_delta(self, spot, T, r, sigma, target_delta, flag) -> float:
        """Numerically invert BSM delta to find the target-delta strike."""
        def obj(K):
            return float(bsm_delta(flag, spot, K, T, r, sigma)) - target_delta
        try:
            K = brentq(obj, spot * 0.30, spot * 2.0, xtol=0.01, maxiter=100)
            return round(K * 2) / 2   # round to nearest $0.50
        except Exception:
            return round(spot)        # fallback: ATM

    def _theo_price(self, flag, spot, K, T, r, sigma, buyer: bool) -> float:
        """
        BSM theoretical price with slip applied to simulate bid/ask spread.
          buyer=True  → pay above theoretical (simulates ask)
          buyer=False → receive below theoretical (simulates bid)
        """
        theo = float(bsm_price(flag, spot, K, T, r, sigma))
        return theo * (1 + self.p.slip) if buyer else theo * (1 - self.p.slip)

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────────

    def _size(self, premium_per_share: float) -> int:
        """Risk at most max_premium_pct of current portfolio value."""
        if premium_per_share <= 0:
            return 0
        pv      = self.broker.getvalue()
        max_usd = pv * self.p.max_premium_pct
        return max(1, int(max_usd / (premium_per_share * 100)))

    # ─────────────────────────────────────────────────────────────────────────
    # Daily position management
    # ─────────────────────────────────────────────────────────────────────────

    def _manage_positions(self, today: date):
        for ticker in list(self.open_trades.keys()):
            meta = self.open_trades[ticker]
            dte  = (meta["expiry"] - today).days

            # Expired without being closed → clean up at zero value
            if dte < 0:
                self.log(f"[{ticker}] expired — recording as loss")
                self._close_trade(ticker, today, reason="expired")
                continue

            # Time exit
            if dte <= self.p.dte_close:
                self.log(f"[{ticker}] {dte} DTE — time exit")
                self._close_trade(ticker, today, reason=f"time_exit_{dte}dte")
                continue

            # Mark-to-market and check profit/stop
            data = self.data_map.get(ticker)
            if data is None:
                continue
            mark = self._compute_mark(meta, data, today)
            if mark is None:
                continue

            prem     = meta["premium"]
            gain_pct = (mark - prem) / prem if prem > 0 else 0.0

            if gain_pct >= (self.p.profit_target - 1.0):
                self.log(f"[{ticker}] profit target +{gain_pct*100:.0f}% — close")
                self._close_trade(ticker, today, reason="profit_target", exit_mark=mark)
            elif gain_pct <= -self.p.stop_loss_pct:
                self.log(f"[{ticker}] stop loss {gain_pct*100:.0f}% — close")
                self._close_trade(ticker, today, reason="stop_loss", exit_mark=mark)

    def _compute_mark(self, meta: dict, data: bt.DataBase,
                      today: date) -> Optional[float]:
        """
        Daily BSM re-price of all legs using updated spot and estimated IV.
        Returns mark per share (same unit as premium).
        """
        spot = float(data.close[0])
        T    = max((meta["expiry"] - today).days / 365, 1 / 365)

        # Update IV estimate; fall back to entry IV if insufficient history
        c, o, h, l = self._read_ohlcv(data, 63)
        rv = _yz_rv(c, o, h, l, window=21) if c is not None else None
        sigma  = (rv * self.p.iv_premium / 100) if rv else meta["entry_sigma"]

        mark = 0.0
        r    = self.p.risk_free
        for action, flag, K in meta["legs"]:
            try:
                # Close: sell long at bid, buy short back at ask
                is_buyer = (action == "buy")
                price    = self._theo_price(flag, spot, K, T, r, sigma,
                                            buyer=(not is_buyer))
                mark += price if action == "buy" else -price
            except Exception:
                return None
        return mark

    def _close_trade(self, ticker: str, today: date,
                     reason: str = "", exit_mark: Optional[float] = None):
        """
        Credit cash for the closing value of a synthetic option position
        and record the trade in trade_log.
        """
        meta = self.open_trades.pop(ticker, None)
        if meta is None:
            return

        data = self.data_map.get(ticker)
        if exit_mark is None and data is not None:
            exit_mark = self._compute_mark(meta, data, today) or 0.0

        exit_mark = exit_mark or 0.0

        n_contracts = meta["n_contracts"]
        n_legs      = len(meta["legs"])
        received    = max(exit_mark, 0) * n_contracts * 100
        commission  = n_legs * n_contracts * self.p.commission
        net_credit  = received - commission

        self.broker.add_cash(net_credit)

        pnl = net_credit - meta["total_cost"]

        record = {
            "ticker":      ticker,
            "trade_type":  meta["trade_type"],
            "direction":   meta["direction"],
            "open_date":   str(meta["open_date"]),
            "close_date":  str(today),
            "entry_spot":  round(meta["entry_spot"], 2),
            "exit_spot":   round(float(data.close[0]), 2) if data else None,
            "premium":     meta["premium"],
            "n_contracts": n_contracts,
            "total_cost":  round(meta["total_cost"], 2),
            "received":    round(net_credit, 2),
            "pnl":         round(pnl, 2),
            "reason":      reason,
        }
        self.trade_log.append(record)
        self.log(
            f"[{ticker}] CLOSE {meta['trade_type']} {meta['direction']} "
            f"reason={reason}  pnl=${pnl:.0f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def log(self, txt: str):
        dt = self.datas[0].datetime.datetime(0)
        log.info("%s | %s", dt.strftime("%Y-%m-%d"), txt)

    def stop(self):
        """Force-close any remaining open positions at end of backtest."""
        today = self.datas[0].datetime.date(0)
        for ticker in list(self.open_trades.keys()):
            self._close_trade(ticker, today, reason="end_of_backtest")
