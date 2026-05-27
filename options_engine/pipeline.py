"""
Full Pipeline — Event Scraper → Vol Signal → Trade Constructor → Probability Engine

Connects all four weeks into a single run:
    pipeline = ShortPremiumPipeline(portfolio_value=100_000)
    results  = pipeline.run("SPY", trade_type="iron_condor")
    results  = pipeline.run_from_event(event)   # driven by Monday scraper
"""
from __future__ import annotations
import logging
from datetime import date, timedelta

from .pricer     import ChainPricer
from .vol_signal import VolSignal
from .constructor import TradeConstructor, Trade
from .probability import ProbabilityEngine, ProbResult

log = logging.getLogger(__name__)


class ShortPremiumPipeline:

    def __init__(
        self,
        portfolio_value:    float = 100_000,
        dte_target:         int   = 30,
        min_pop:            float = 60.0,    # skip trade if MC POP below this
        mc_model:           str   = "student_t",
        mc_sims:            int   = 50_000,
    ):
        self.portfolio_value = portfolio_value
        self.dte_target      = dte_target
        self.min_pop         = min_pop
        self.mc_model        = mc_model
        self.mc_sims         = mc_sims

        self.constructor = TradeConstructor(portfolio_value=portfolio_value)
        self.prob_engine = ProbabilityEngine()

    # ── Main entry points ─────────────────────────────────────────────────

    def run(
        self,
        ticker:     str,
        trade_type: str = "iron_condor",   # "iron_condor" | "strangle" | "straddle"
        wing_width: float = 15.0,
    ) -> dict:
        """
        Full pipeline for a single ticker.
        Returns a result dict with trade, prob, and go/no-go decision.
        """
        log.info("[%s] Step 1/4 — Fetching chain & pricing Greeks …", ticker)
        pricer = ChainPricer(ticker)
        chain  = pricer.price_chain(dte_target=self.dte_target)

        log.info("[%s] Step 2/4 — Computing vol signal (IV rank + VRP) …", ticker)
        signal = VolSignal(ticker).evaluate()

        # Go / no-go on vol signal
        if signal["signal"] != "SELL":
            log.info("[%s] Vol signal is NEUTRAL — skipping trade construction", ticker)
            return {"ticker": ticker, "decision": "PASS", "reason": "vol_signal_neutral",
                    "signal": signal, "trade": None, "prob": None}

        log.info("[%s] Step 3/4 — Constructing %s …", ticker, trade_type)
        build_fn = {
            "iron_condor": lambda: self.constructor.iron_condor(chain, pricer, signal=signal, wing_width=wing_width),
            "strangle":    lambda: self.constructor.strangle(chain, pricer, signal=signal),
            "straddle":    lambda: self.constructor.straddle(chain, pricer, signal=signal),
        }.get(trade_type)
        if not build_fn:
            raise ValueError(f"Unknown trade_type: {trade_type}")
        trade = build_fn()

        log.info("[%s] Step 4/4 — Running Monte Carlo (%s, n=%d) …",
                 ticker, self.mc_model, self.mc_sims)
        prob = self.prob_engine.evaluate(trade, model=self.mc_model, n_sims=self.mc_sims)

        # Final go / no-go on POP
        decision = "TRADE" if prob.pop >= self.min_pop else "PASS"
        if decision == "PASS":
            log.info("[%s] MC POP=%.1f%% below threshold %.1f%% — skipping", ticker, prob.pop, self.min_pop)

        # Update contract count with MC-derived POP
        trade.n_contracts = self.constructor.kelly_size(
            prob.pop, trade.net_credit, trade.max_loss
        )
        trade.capital_at_risk = round(trade.max_loss * trade.n_contracts * 100, 2)
        trade.portfolio_pct   = round(trade.capital_at_risk / self.portfolio_value * 100, 2)

        return {
            "ticker":   ticker,
            "decision": decision,
            "reason":   f"pop={prob.pop:.1f}%",
            "signal":   signal,
            "trade":    trade,
            "prob":     prob,
        }

    def run_from_event(self, event: dict, trade_type: str = "iron_condor") -> dict | None:
        """
        Driven by the Monday event scraper.
        event dict must have 'ticker', 'event_type', 'date', 'days_until'.
        Only constructs trades for events within the DTE window.
        """
        ticker = event.get("ticker")
        if not ticker:
            return None

        days = event.get("days_until", 0)
        if not (5 <= days <= self.dte_target + 5):
            log.debug("Event %s: %d DTE outside window — skip", ticker, days)
            return None

        log.info("Running pipeline for event: %s on %s (%d DTE)",
                 event.get("title"), event.get("event_date"), days)
        return self.run(ticker, trade_type=trade_type)

    def scan_watchlist(
        self,
        tickers:    list[str],
        trade_type: str = "iron_condor",
    ) -> list[dict]:
        """
        Scan a list of tickers and return all TRADE decisions ranked by EV.
        """
        results = []
        for ticker in tickers:
            try:
                r = self.run(ticker, trade_type=trade_type)
                if r["decision"] == "TRADE":
                    results.append(r)
            except Exception as e:
                log.warning("Pipeline failed for %s: %s", ticker, e)

        # Rank by expected value descending
        results.sort(key=lambda r: r["prob"].ev if r["prob"] else 0, reverse=True)
        return results
