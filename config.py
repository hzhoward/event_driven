"""
Central config — API keys go in environment variables, never committed.
"""
import os
from datetime import datetime, timedelta
import pytz

# ─── Timezone ────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")

# ─── Lookahead window ────────────────────────────────────────────────────────
LOOKAHEAD_DAYS = 14   # Monday briefing covers next two weeks

# ─── API keys (set in env) ────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")          # https://fred.stlouisfed.org/docs/api/api_key.html
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY", "")  # optional enrichment

# ─── Impact scoring weights ───────────────────────────────────────────────────
# Each event type gets a base impact score 1-10
IMPACT_WEIGHTS = {
    "earnings_mega_cap":   9,   # AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA
    "earnings_large_cap":  7,
    "earnings_mid_cap":    5,
    "earnings_small_cap":  3,
    "fomc_decision":      10,
    "fomc_minutes":        7,
    "fed_speech_chair":    8,
    "fed_speech_other":    4,
    "cpi":                 9,
    "ppi":                 7,
    "nfp":                 9,   # non-farm payrolls
    "gdp":                 8,
    "retail_sales":        6,
    "ism_manufacturing":   6,
    "ism_services":        6,
    "jobless_claims":      5,
    "housing_starts":      4,
    "fda_pdufa":           9,   # binary drug approval
    "fda_adcom":           8,   # advisory committee (leading indicator)
    "ma_announcement":     8,
    "index_rebalance":     6,
    "investor_day":        6,
    "analyst_day":         5,
}

# ─── Mega-cap universe (straddle candidates around earnings) ─────────────────
MEGA_CAP_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META",
    "TSLA", "BRK.B", "UNH", "JPM", "V", "XOM", "LLY", "JNJ",
    "AVGO", "MA", "PG", "HD", "MRK", "COST", "ABBV", "CVX", "BAC",
    "NFLX", "AMD", "CRM", "ORCL", "ACN", "TMO", "PEP", "ADBE",
    "MCD", "CSCO", "WMT", "KO", "GE", "DIS", "IBM", "GS", "MS",
    "SPY", "QQQ", "IWM",  # index ETFs
}

# ─── Strategy decision matrix ────────────────────────────────────────────────
# Maps (event_type, iv_rank_bucket) → recommended strategy
# iv_rank_bucket: "low" <25th pct, "med" 25-75th, "high" >75th pct
STRATEGY_MATRIX = {
    # Earnings: buy vol before (straddle/strangle), sell after crush
    ("earnings", "low"):  {
        "pre":  "Long Straddle",
        "post": "Short Straddle",
        "note": "IV likely to expand into print; fade crush post-announcement",
    },
    ("earnings", "med"):  {
        "pre":  "Long Strangle (OTM 1σ)",
        "post": "Iron Condor",
        "note": "Balanced risk; strangle cheaper entry than straddle",
    },
    ("earnings", "high"): {
        "pre":  "Sell Put / Bull Put Spread",
        "post": "Short Straddle",
        "note": "IV already elevated; premium selling more attractive",
    },
    # Macro binary events
    ("fomc_decision", "low"):  {"pre": "Long Straddle SPY/QQQ", "post": "Close", "note": "Cheap vol, big potential move"},
    ("fomc_decision", "med"):  {"pre": "Long Strangle SPY/QQQ", "post": "Close", "note": "Standard playbook"},
    ("fomc_decision", "high"): {"pre": "Iron Condor SPY",       "post": "Close", "note": "Sell the vol spike"},
    ("cpi", "low"):            {"pre": "Long Straddle SPY",     "post": "Close", "note": "Cheap entry, surprise potential high"},
    ("cpi", "med"):            {"pre": "Long Strangle SPY",     "post": "Close", "note": ""},
    ("cpi", "high"):           {"pre": "Iron Condor SPY",       "post": "Close", "note": "Rich premium"},
    ("nfp", "low"):            {"pre": "Long Straddle SPY",     "post": "Close", "note": ""},
    ("nfp", "med"):            {"pre": "Long Strangle SPY",     "post": "Close", "note": ""},
    ("nfp", "high"):           {"pre": "Iron Condor SPY",       "post": "Close", "note": ""},
    # FDA binary
    ("fda_pdufa", "low"):      {"pre": "Long Straddle",         "post": "Close", "note": "50/50 binary; IV usually cheap early"},
    ("fda_pdufa", "med"):      {"pre": "Long Straddle",         "post": "Close", "note": "Classic binary play"},
    ("fda_pdufa", "high"):     {"pre": "Risk Reversal",         "post": "Close", "note": "IV too rich for debit; use skew"},
    # M&A
    ("ma_announcement", "low"): {"pre": "Long Call (target)",  "post": "Hold",  "note": "Merger arb; call on target captures upside"},
    ("ma_announcement", "med"): {"pre": "Long Call (target)",  "post": "Hold",  "note": ""},
    ("ma_announcement", "high"):{"pre": "Risk Reversal",       "post": "Hold",  "note": ""},
}

# ─── Storage ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "events.db")

# ─── Briefing output ─────────────────────────────────────────────────────────
BRIEFING_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "briefings")
