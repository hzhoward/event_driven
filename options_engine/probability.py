"""
Week 4 — Probability Engine (Monte Carlo + Analytical)

Two modes:
  analytical  : closed-form BSM POP (fast, ~1ms, assumes log-normal)
  monte_carlo : fat-tail simulation (Student-t ν=4 + optional Merton jumps)

Short-premium PM use cases:
  - POP > 60% : go signal for trade entry
  - E[P&L]    : risk-adjusted expected value per contract
  - CVaR 95%  : expected loss in the worst 5% of outcomes
  - Scenario  : P&L surface across spot moves + vol changes
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t

from .pricer import bsm_price
from .constructor import Trade


@dataclass
class ProbResult:
    # Core metrics
    pop:         float   # probability of profit (%)
    ev:          float   # expected value per contract ($)
    cvar_95:     float   # conditional VaR at 95% ($, negative = loss)

    # Distribution
    p10: float; p25: float; p50: float; p75: float; p90: float  # price percentiles at expiry

    # Model details
    model:       str     # "analytical", "gbm", "student_t", "merton"
    n_sims:      int
    df_used:     float | None   # Student-t degrees of freedom

    def show(self):
        print(
            f"\n  Model: {self.model}  (n={self.n_sims:,})\n"
            f"  POP:    {self.pop:.1f}%\n"
            f"  E[P&L]: ${self.ev:,.2f} per contract\n"
            f"  CVaR95: ${self.cvar_95:,.2f} per contract\n"
            f"  Price at expiry — p10:${self.p10:.1f}  p25:${self.p25:.1f}  "
            f"p50:${self.p50:.1f}  p75:${self.p75:.1f}  p90:${self.p90:.1f}\n"
        )


class ProbabilityEngine:
    """
    Evaluates a Trade's probability of profit and expected value
    under three progressively realistic market models.
    """

    def __init__(self, risk_free: float = 0.053):
        self.risk_free = risk_free

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        trade: Trade,
        model: str = "student_t",    # "analytical" | "gbm" | "student_t" | "merton"
        n_sims: int = 50_000,
        student_df: float = 4.0,     # degrees of freedom; lower = fatter tails
        jump_intensity: float = 1.0, # Merton: expected jumps per year
        jump_mean: float = -0.02,    # Merton: mean jump size (negative = crash bias)
        jump_vol: float = 0.04,      # Merton: jump size std dev
    ) -> ProbResult:
        """
        Run the probability engine on a constructed trade.
        Updates trade.n_contracts using the MC-derived POP (replaces placeholder).
        """
        iv    = self._trade_iv(trade)
        T     = trade.dte / 365
        S     = trade.spot
        lower = trade.breakeven_lower
        upper = trade.breakeven_upper

        if model == "analytical":
            return self._analytical(trade, S, T, iv, lower, upper)

        # Simulate terminal prices
        S_T = self._simulate(model, S, T, iv, n_sims,
                             student_df, jump_intensity, jump_mean, jump_vol)

        pop   = float(np.mean((S_T >= lower) & (S_T <= upper)) * 100)
        pnl   = self._pnl_per_contract(trade, S_T)
        ev    = float(np.mean(pnl))
        cvar  = float(np.mean(pnl[pnl <= np.percentile(pnl, 5)]))
        pcts  = np.percentile(S_T, [10, 25, 50, 75, 90])

        return ProbResult(
            pop=round(pop, 1), ev=round(ev, 2), cvar_95=round(cvar, 2),
            p10=round(pcts[0], 2), p25=round(pcts[1], 2), p50=round(pcts[2], 2),
            p75=round(pcts[3], 2), p90=round(pcts[4], 2),
            model=model, n_sims=n_sims, df_used=student_df if "student" in model else None,
        )

    def scenario_grid(
        self, trade: Trade,
        spot_moves:  list[float] = [-0.10, -0.05, 0, 0.05, 0.10],
        vol_changes: list[float] = [-0.05, 0, 0.05],
    ) -> "pd.DataFrame":
        """
        P&L grid across spot % moves and IV shifts.
        Useful for visualising gamma/vega risk before entry.
        """
        import pandas as pd
        iv   = self._trade_iv(trade)
        T    = trade.dte / 365
        rows = []
        for dv in vol_changes:
            row = {"vol_chg": f"{dv:+.0%}"}
            for ds in spot_moves:
                S_new     = trade.spot * (1 + ds)
                iv_new    = max(0.01, iv + dv)
                pnl_share = self._instant_pnl(trade, S_new, iv_new, T)
                row[f"spot{ds:+.0%}"] = round(pnl_share * 100, 0)  # per contract
            rows.append(row)
        return pd.DataFrame(rows).set_index("vol_chg")

    # ── Simulation models ─────────────────────────────────────────────────

    def _simulate(self, model, S, T, sigma, n, df, lam, mu_j, sig_j) -> np.ndarray:
        """Returns array of n terminal prices under the chosen model."""
        rng = np.random.default_rng(seed=42)   # reproducible

        if model == "gbm":
            # Standard log-normal (Black-Scholes world)
            z   = rng.standard_normal(n)
            return S * np.exp((self.risk_free - 0.5 * sigma**2) * T
                              + sigma * np.sqrt(T) * z)

        elif model == "student_t":
            # Student-t: same drift, fatter tails — more realistic for equities
            # Scale t-draws to match σ (variance = df/(df-2) for df>2)
            z   = rng.standard_t(df, size=n)
            z  *= np.sqrt((df - 2) / df)         # rescale to unit variance
            return S * np.exp((self.risk_free - 0.5 * sigma**2) * T
                              + sigma * np.sqrt(T) * z)

        elif model == "merton":
            # Merton (1976) jump-diffusion: GBM + compound Poisson jumps
            # λ = jump intensity, J ~ N(μ_j, σ_j²)
            k   = np.exp(mu_j + 0.5 * sig_j**2) - 1   # mean jump size
            adj = self.risk_free - lam * k - 0.5 * sigma**2   # drift adjustment

            z      = rng.standard_normal(n)
            n_jump = rng.poisson(lam * T, n)             # number of jumps per path
            j_size = np.array([
                np.sum(rng.normal(mu_j, sig_j, nj)) if nj > 0 else 0.0
                for nj in n_jump
            ])
            return S * np.exp(adj * T + sigma * np.sqrt(T) * z + j_size)

        raise ValueError(f"Unknown model: {model}")

    # ── Analytical (closed-form) ──────────────────────────────────────────

    def _analytical(self, trade, S, T, sigma, lower, upper) -> ProbResult:
        """
        BSM-based POP: P(lower < S_T < upper) = N(d2_upper) - N(d2_lower)
        Fast (no simulation), assumes log-normal.
        """
        r = self.risk_free

        def d2(K):
            return (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        pop = (norm.cdf(d2(lower)) - norm.cdf(d2(upper))) * 100  # put side prob
        # For a strangle/IC: P(profit) = P(lower < S < upper)
        pop = abs(float(norm.cdf(d2(upper)) - norm.cdf(d2(lower))) * 100)

        # EV approximation: credit × POP - max_loss × (1-POP)
        p = pop / 100
        ev = (trade.net_credit * p - trade.max_loss * (1 - p)) * 100

        # Percentiles from log-normal
        def lognorm_pct(q):
            return S * np.exp((r - 0.5 * sigma**2) * T
                              + sigma * np.sqrt(T) * norm.ppf(q))

        return ProbResult(
            pop=round(pop, 1), ev=round(ev, 2), cvar_95=round(-trade.max_loss * 100, 2),
            p10=round(lognorm_pct(0.10), 2), p25=round(lognorm_pct(0.25), 2),
            p50=round(lognorm_pct(0.50), 2), p75=round(lognorm_pct(0.75), 2),
            p90=round(lognorm_pct(0.90), 2),
            model="analytical", n_sims=0, df_used=None,
        )

    # ── P&L calculation ───────────────────────────────────────────────────

    def _pnl_per_contract(self, trade: Trade, S_T: np.ndarray) -> np.ndarray:
        """
        P&L at expiry for each simulated path.
        Short legs expire worthless (full credit) or are assigned.
        Long legs offset losses beyond wing strikes.
        """
        pnl = np.zeros(len(S_T))
        for leg in trade.legs:
            K, flag = leg.strike, leg.flag
            intrinsic = np.maximum(S_T - K, 0) if flag == "c" else np.maximum(K - S_T, 0)
            if leg.action == "sell":
                pnl += (leg.mid - intrinsic)    # collected premium minus assignment
            else:
                pnl += (intrinsic - leg.mid)    # long wing payoff minus cost
        return pnl * 100   # per contract (100 shares)

    def _instant_pnl(self, trade: Trade, S_new: float, iv_new: float, T_remaining: float) -> float:
        """Mark-to-market P&L per share using BSM repricing."""
        pnl = 0.0
        for leg in trade.legs:
            if T_remaining <= 0:
                intrinsic = max(S_new - leg.strike, 0) if leg.flag == "c" else max(leg.strike - S_new, 0)
                new_price = intrinsic
            else:
                new_price = float(bsm_price(
                    leg.flag, S_new, leg.strike, T_remaining, self.risk_free, iv_new
                ))
            pnl += (leg.mid - new_price) if leg.action == "sell" else (new_price - leg.mid)
        return pnl

    # ── Helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _trade_iv(trade: Trade) -> float:
        """Use average IV of short legs as the simulation vol."""
        short_ivs = [l.iv for l in trade.legs if l.action == "sell" and l.iv and l.iv > 0]
        return float(np.mean(short_ivs)) / 100 if short_ivs else 0.18
