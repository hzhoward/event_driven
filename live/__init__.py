"""
live/ — Alpaca paper-trading live execution layer.

Architecture
------------
alpaca_client.py   Low-level Alpaca REST wrapper (chain lookup, order routing,
                   account/position queries).
signal.py          Reusable signal helpers shared by trader and backtester:
                   Yang-Zhang RV, IV rank, momentum direction, BSM delta strike.
event_trader.py    Monday evaluation: scraper events → vol signal → Alpaca orders.
position_manager.py  Daily 15:45 ET check: profit target / stop-loss / DTE exits.
run.py             CLI entry point + scheduler.

Credentials are loaded exclusively from environment variables (never hardcoded).
Create a .env file in the project root (see .env.example):

    ALPACA_API_KEY=<your_key>
    ALPACA_SECRET_KEY=<your_secret>
    ALPACA_PAPER=true          # set to false for live trading
"""
from __future__ import annotations
