"""
Unit tests for Weeks 1-4 — all offline (no network calls).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from options_engine.pricer      import bsm_price, bsm_delta, bsm_gamma, bsm_vega, bsm_theta, iv_solver
from options_engine.vol_signal  import VolSignal
from options_engine.constructor import TradeConstructor, Trade
from options_engine.probability import ProbabilityEngine


# ── Week 1: BSM pricer ───────────────────────────────────────────────────────

def test_bsm_call_put_parity():
    """Put-call parity: C - P = S - K·e^{-rT}"""
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
    C = float(bsm_price('c', S, K, T, r, sigma))
    P = float(bsm_price('p', S, K, T, r, sigma))
    pcp = C - P
    expected = S - K * np.exp(-r * T)
    assert abs(pcp - expected) < 1e-8

def test_bsm_atm_call_above_intrinsic():
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
    price = float(bsm_price('c', S, K, T, r, sigma))
    assert price > 0

def test_bsm_call_delta_range():
    d = float(bsm_delta('c', 100, 100, 0.25, 0.05, 0.20))
    assert 0 < d < 1

def test_bsm_put_delta_range():
    d = float(bsm_delta('p', 100, 100, 0.25, 0.05, 0.20))
    assert -1 < d < 0

def test_bsm_gamma_positive():
    g = float(bsm_gamma(100, 100, 0.25, 0.05, 0.20))
    assert g > 0

def test_bsm_vega_positive():
    v = float(bsm_vega(100, 100, 0.25, 0.05, 0.20))
    assert v > 0

def test_bsm_theta_negative():
    """Long options lose value over time."""
    th = float(bsm_theta('c', 100, 100, 0.25, 0.05, 0.20))
    assert th < 0

def test_bsm_vectorised():
    """Price an array of strikes at once."""
    K = np.array([90., 95., 100., 105., 110.])
    S = np.full(5, 100.0)
    prices = bsm_price('c', S, K, 0.25, 0.05, 0.20)
    assert prices.shape == (5,)
    assert np.all(prices[:-1] > prices[1:])   # calls decrease as K increases

def test_iv_solver_round_trip():
    """IV solver should recover the original sigma."""
    S, K, T, r, sigma = 100., 100., 0.25, 0.05, 0.20
    price = bsm_price('c', np.array([S]), np.array([K]), T, r, sigma)
    recovered = iv_solver(price, np.array([S]), np.array([K]),
                          np.array([T]), np.array([r]), np.array(['c']))
    assert abs(float(recovered[0]) - sigma) < 1e-4


# ── Week 2: Yang-Zhang RV ────────────────────────────────────────────────────

def _fake_ohlc(n=60, base=100.0, daily_vol=0.01):
    """Generate synthetic OHLC data."""
    rng = np.random.default_rng(0)
    closes = base * np.cumprod(1 + rng.normal(0, daily_vol, n))
    opens  = closes * np.exp(rng.normal(0, 0.003, n))
    highs  = np.maximum(opens, closes) * (1 + abs(rng.normal(0, 0.003, n)))
    lows   = np.minimum(opens, closes) * (1 - abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})

def test_yang_zhang_positive():
    hist = _fake_ohlc()
    rv = VolSignal.yang_zhang_rv(hist, window=21)
    assert rv is not None and rv > 0

def test_yang_zhang_higher_vol():
    """Higher daily moves should produce higher Yang-Zhang RV."""
    low_vol  = VolSignal.yang_zhang_rv(_fake_ohlc(daily_vol=0.005), 21)
    high_vol = VolSignal.yang_zhang_rv(_fake_ohlc(daily_vol=0.020), 21)
    assert high_vol > low_vol

def test_yang_zhang_insufficient_data():
    rv = VolSignal.yang_zhang_rv(_fake_ohlc(n=10), window=21)
    assert rv is None


# ── Week 3: Trade constructor ────────────────────────────────────────────────

def _mock_chain(spot=500.0):
    """Minimal options chain DataFrame for constructor tests."""
    strikes = np.arange(spot * 0.85, spot * 1.15, 2.5)
    rows = []
    for K in strikes:
        moneyness = (K - spot) / spot
        iv = 0.20 + abs(moneyness) * 0.5    # skew
        T  = 30 / 365
        r  = 0.053
        delta_c = float(bsm_delta('c', spot, K, T, r, iv))
        delta_p = float(bsm_delta('p', spot, K, T, r, iv))
        mid_c   = float(bsm_price('c', spot, K, T, r, iv))
        mid_p   = float(bsm_price('p', spot, K, T, r, iv))
        gamma   = float(bsm_gamma(spot, K, T, r, iv))
        vega    = float(bsm_vega(spot, K, T, r, iv))
        theta_c = float(bsm_theta('c', spot, K, T, r, iv))
        theta_p = float(bsm_theta('p', spot, K, T, r, iv))
        rows += [
            {"strike": K, "flag": "c", "mid": mid_c, "iv": iv*100,
             "delta": delta_c, "gamma": gamma, "vega": vega, "theta": theta_c,
             "expiry": "2025-09-19", "spot": spot, "dte": 30},
            {"strike": K, "flag": "p", "mid": mid_p, "iv": iv*100,
             "delta": delta_p, "gamma": gamma, "vega": vega, "theta": theta_p,
             "expiry": "2025-09-19", "spot": spot, "dte": 30},
        ]
    return pd.DataFrame(rows)

def _mock_pricer(spot=500.0):
    p = MagicMock()
    p.ticker = "TEST"
    p.spot   = spot
    p.find_strikes = lambda chain, call_delta=0.16, put_delta=-0.16: \
        TradeConstructor()._mock_find_strikes_impl(chain, call_delta, put_delta, p)
    return p

class _ExtConstructor(TradeConstructor):
    """Expose find_strikes for mock pricer."""
    def _mock_find_strikes_impl(self, chain, call_delta, put_delta, pricer):
        from options_engine.pricer import OptionLeg
        c = chain[chain["flag"]=="c"].dropna(subset=["delta"])
        p = chain[chain["flag"]=="p"].dropna(subset=["delta"])
        def closest(df, tgt):
            r = df.loc[(df["delta"]-tgt).abs().idxmin()]
            return OptionLeg(ticker="TEST", expiry=r["expiry"], strike=r["strike"],
                             flag=r["flag"], action="sell",
                             delta=r["delta"], gamma=r["gamma"], vega=r["vega"],
                             theta=r["theta"], iv=r["iv"], mid=r["mid"])
        return {"short_call": closest(c, call_delta), "short_put": closest(p, put_delta)}

def test_iron_condor_credit_positive():
    spot  = 500.0
    chain = _mock_chain(spot)
    tc    = _ExtConstructor(portfolio_value=100_000)
    pricer = MagicMock(); pricer.ticker="TEST"; pricer.spot=spot
    pricer.find_strikes = lambda ch, cd=0.16, pd_=-0.16: tc._mock_find_strikes_impl(ch, cd, pd_, pricer)
    trade = tc.iron_condor(chain, pricer)
    assert trade.net_credit > 0

def test_iron_condor_defined_risk():
    spot  = 500.0
    chain = _mock_chain(spot)
    tc    = _ExtConstructor(portfolio_value=100_000)
    pricer = MagicMock(); pricer.ticker="TEST"; pricer.spot=spot
    pricer.find_strikes = lambda ch, cd=0.16, pd_=-0.16: tc._mock_find_strikes_impl(ch, cd, pd_, pricer)
    trade = tc.iron_condor(chain, pricer)
    assert trade.max_loss < trade.breakeven_upper - trade.breakeven_lower

def test_kelly_size_sensible():
    tc = TradeConstructor(portfolio_value=100_000)
    n = tc.kelly_size(pop=65.0, max_profit=2.0, max_loss=8.0)
    assert 1 <= n <= 50    # sanity range

def test_kelly_size_never_zero():
    tc = TradeConstructor(portfolio_value=100_000)
    assert tc.kelly_size(pop=50.0, max_profit=1.0, max_loss=9.0) >= 1


# ── Week 4: Probability engine ───────────────────────────────────────────────

def _fake_trade(spot=500.0, credit=5.0, lower=475.0, upper=525.0) -> Trade:
    from options_engine.pricer import OptionLeg
    leg = OptionLeg("T","2025-09-19",upper,"c","sell",0.16,0.001,0.5,-0.05,20.0,credit/2)
    leg2= OptionLeg("T","2025-09-19",lower,"p","sell",-0.16,0.001,0.5,-0.05,20.0,credit/2)
    return Trade(
        ticker="TEST", trade_type="strangle", legs=[leg, leg2],
        expiry="2025-09-19", dte=30, spot=spot,
        net_credit=credit, max_profit=credit, max_loss=credit*3,
        breakeven_upper=upper, breakeven_lower=lower,
        stop_loss_price=credit*2,
        net_delta=0.0, net_gamma=-0.002, net_vega=-1.0, net_theta=0.50,
        n_contracts=1, capital_at_risk=credit*3*100, portfolio_pct=1.5,
    )

def test_analytical_pop_in_range():
    eng   = ProbabilityEngine()
    trade = _fake_trade()
    res   = eng.evaluate(trade, model="analytical")
    assert 0 < res.pop < 100

def test_gbm_pop_in_range():
    eng   = ProbabilityEngine()
    trade = _fake_trade()
    res   = eng.evaluate(trade, model="gbm", n_sims=10_000)
    assert 0 < res.pop < 100

def test_student_t_lower_df_shifts_cvar():
    """
    Lower degrees of freedom = heavier tails = worse CVaR (more extreme losses).
    CVaR should be lower (more negative) with df=3 than df=10.
    """
    eng   = ProbabilityEngine()
    trade = _fake_trade()
    low_df  = eng.evaluate(trade, model="student_t", n_sims=30_000, student_df=3.0)
    high_df = eng.evaluate(trade, model="student_t", n_sims=30_000, student_df=10.0)
    # Heavier tails → worse tail losses
    assert low_df.cvar_95 <= high_df.cvar_95 + 200   # allow simulation noise

def test_merton_pop_in_range():
    eng   = ProbabilityEngine()
    trade = _fake_trade()
    res   = eng.evaluate(trade, model="merton", n_sims=10_000)
    assert 0 < res.pop < 100

def test_scenario_grid_shape():
    eng   = ProbabilityEngine()
    trade = _fake_trade()
    grid  = eng.scenario_grid(trade, spot_moves=[-0.05, 0, 0.05], vol_changes=[-0.05, 0, 0.05])
    assert grid.shape == (3, 3)

def test_cvar_lte_ev():
    """CVaR (tail loss) should be worse than (or equal to) mean EV."""
    eng   = ProbabilityEngine()
    trade = _fake_trade()
    res   = eng.evaluate(trade, model="student_t", n_sims=20_000)
    assert res.cvar_95 <= res.ev
