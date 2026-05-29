"""
Week 1 — Chain Pricer & Greeks  (scipy BSM, fully vectorised)

Prices a full options chain and computes all Greeks in one numpy call.
Uses scipy.stats.norm — no numba, no extra dependencies, QC-portable.

Usage:
    pricer = ChainPricer("SPY")
    chain  = pricer.price_chain(dte_target=30)   # DataFrame: price + greeks
    legs   = pricer.find_strikes(chain)           # 16Δ short call + put
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

RISK_FREE = 0.053   # update periodically or pull from FRED


# ── Core BSM (vectorised) ─────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bsm_price(flag, S, K, T, r, sigma):
    """Vectorised BSM price. flag: 'c' or 'p' (or arrays of each)."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(np.asarray(flag) == 'c', call, put)

def bsm_delta(flag, S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    call_d = norm.cdf(d1)
    put_d  = call_d - 1
    return np.where(np.asarray(flag) == 'c', call_d, put_d)

def bsm_gamma(S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bsm_vega(S, K, T, r, sigma):
    """Vega per 1% change in IV."""
    d1, _ = _d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def bsm_theta(flag, S, K, T, r, sigma):
    """Theta per calendar day."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    call_t = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    put_t  = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return np.where(np.asarray(flag) == 'c', call_t, put_t) / 365

def iv_solver(market_price, S, K, T, r, flag, tol=1e-6, max_iter=100):
    """
    Vectorised Newton-Raphson IV solver.
    Returns NaN for invalid inputs (deep ITM/OTM with no vol solution).
    """
    sigma = np.full_like(market_price, 0.3, dtype=float)   # initial guess
    for _ in range(max_iter):
        price = bsm_price(flag, S, K, T, r, sigma)
        vega  = bsm_vega(S, K, T, r, sigma) * 100          # undo /100
        diff  = price - market_price
        safe_vega = np.where(np.abs(vega) > 1e-10, vega, np.inf)
        step  = diff / safe_vega
        sigma -= step
        sigma  = np.clip(sigma, 1e-4, 5.0)
        if np.max(np.abs(diff)) < tol:
            break
    # Mask solutions that didn't converge
    final_price = bsm_price(flag, S, K, T, r, sigma)
    sigma = np.where(np.abs(final_price - market_price) < 0.05, sigma, np.nan)
    return sigma


# ── Chain Pricer ──────────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    """Single option leg — feeds into trade constructor (Week 3)."""
    ticker: str;  expiry: str;  strike: float;  flag: str;  action: str
    delta: float; gamma: float; vega: float;    theta: float
    iv: float;    mid: float


class ChainPricer:
    """
    Prices the full options chain for a ticker with Greeks.
    Strike selection uses delta targeting for the trade constructor.
    """

    def __init__(self, ticker: str, risk_free: float = RISK_FREE):
        self.ticker    = ticker.upper()
        self.risk_free = risk_free
        self._yf       = yf.Ticker(self.ticker)
        self.spot      = float(self._yf.history(period="1d")["Close"].iloc[-1])

    def price_chain(self, dte_target: int = 30) -> pd.DataFrame:
        """Full chain DataFrame: strike, flag, mid, iv, delta, gamma, vega, theta."""
        expiry = self._nearest_expiry(dte_target)
        T = (date.fromisoformat(expiry) - date.today()).days / 365
        if T <= 0:
            raise ValueError(f"Expiry {expiry} is in the past")

        chain_raw = self._yf.option_chain(expiry)
        df = pd.concat([
            self._price_side(chain_raw.calls, "c", T),
            self._price_side(chain_raw.puts,  "p", T),
        ], ignore_index=True)
        df["expiry"] = expiry
        df["spot"]   = self.spot
        df["dte"]    = round(T * 365)
        return df.sort_values(["flag", "strike"]).reset_index(drop=True)

    def find_strikes(self, df: pd.DataFrame,
                     call_delta: float = 0.16,
                     put_delta:  float = -0.16) -> dict[str, OptionLeg]:
        """Returns the legs closest to the target deltas (default: 16Δ wings)."""
        return {
            "short_call": self._closest(df[df["flag"] == "c"], call_delta),
            "short_put":  self._closest(df[df["flag"] == "p"], put_delta),
        }

    def iv_surface(self, dtes: list[int] = [7, 14, 30, 45, 60]) -> pd.DataFrame:
        """ATM IV term structure — used by VolSignal for IV rank."""
        rows = []
        for dte in dtes:
            try:
                exp = self._nearest_expiry(dte)
                T   = (date.fromisoformat(exp) - date.today()).days / 365
                if T <= 0:
                    continue
                c   = self._yf.option_chain(exp).calls
                atm = c.iloc[(c["strike"] - self.spot).abs().argsort()[:1]]
                mid = float((atm["bid"] + atm["ask"]).values[0] / 2)
                iv  = float(iv_solver(
                    np.array([mid]), np.array([self.spot]),
                    atm["strike"].values, np.array([T]),
                    np.array([self.risk_free]), np.array(["c"])
                )[0])
                rows.append({"dte": round(T * 365), "expiry": exp, "atm_iv": round(iv * 100, 2)})
            except Exception:
                pass
        return pd.DataFrame(rows)

    # ── Internals ─────────────────────────────────────────────────────────

    def _price_side(self, chain: pd.DataFrame, flag: str, T: float) -> pd.DataFrame:
        df   = chain[["strike", "bid", "ask"]].copy().reset_index(drop=True)
        mid  = ((df["bid"] + df["ask"]) / 2).values
        K    = df["strike"].values
        S, r = np.full(len(K), self.spot), np.full(len(K), self.risk_free)
        t    = np.full(len(K), T)
        flags= np.array([flag] * len(K))

        iv    = iv_solver(mid, S, K, t, r, flags)
        valid = np.isfinite(iv)

        return pd.DataFrame({
            "strike": K,
            "flag":   flag,
            "mid":    np.round(mid, 2),
            "iv":     np.where(valid, np.round(iv * 100, 2), np.nan),
            "delta":  np.where(valid, np.round(bsm_delta(flags, S, K, t, r, iv), 4), np.nan),
            "gamma":  np.where(valid, np.round(bsm_gamma(S, K, t, r, iv),        6), np.nan),
            "vega":   np.where(valid, np.round(bsm_vega(S, K, t, r, iv),         4), np.nan),
            "theta":  np.where(valid, np.round(bsm_theta(flags, S, K, t, r, iv), 4), np.nan),
        })

    def _nearest_expiry(self, dte_target: int) -> str:
        exps = self._yf.options
        if not exps:
            raise ValueError(f"No options data for {self.ticker}")
        today = date.today()
        return min(exps, key=lambda e: abs((date.fromisoformat(e) - today).days - dte_target))

    def _closest(self, df: pd.DataFrame, target_delta: float) -> OptionLeg:
        df = df.dropna(subset=["delta"])
        r  = df.loc[(df["delta"] - target_delta).abs().idxmin()]
        return OptionLeg(
            ticker=self.ticker, expiry=r["expiry"], strike=r["strike"],
            flag=r["flag"], action="sell",
            delta=r["delta"], gamma=r["gamma"], vega=r["vega"], theta=r["theta"],
            iv=r["iv"], mid=r["mid"],
        )
