"""
Backtrader backtest runner — Long-Gamma/Vega strategy.

Usage:
    # From the project root (event_driven/):
    python -m backtrader.run

    # Custom date range and cash:
    python -m backtrader.run --start 2023-01-01 --end 2024-12-31 --cash 200000

    # Single ticker for quick testing:
    python -m backtrader.run --tickers SPY QQQ

CLI args:
    --start   YYYY-MM-DD  (default 2022-01-01)
    --end     YYYY-MM-DD  (default 2024-12-31)
    --cash    initial portfolio value (default 100000)
    --tickers space-separated list  (overrides built-in WATCHLIST)
    --plot    show backtrader equity chart after run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Make sure project root is on the path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtrader as bt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from bt_engine.strategy   import LongGammaStrategy
from bt_engine.data_feed  import load_feeds
from bt_engine.commission import OptionsCommission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log     = logging.getLogger(__name__)
console = Console()

# ── Default watchlist (same as algorithm_qc.py) ───────────────────────────────

WATCHLIST = [
    "SPY",  "QQQ",  "IWM",
    "NVDA", "AMD",  "AVGO", "MSFT", "META",
    "PLTR", "ANET", "VRT",  "GEV",  "CRDO",
    "TSM",  "ORCL", "PANW", "NOW",  "ASML",
    "AMAT", "NFLX", "NEE",  "EQIX",
    "DLR",  "AMT",  "TSLA", "WDC",  "INTC",
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_backtest(
    tickers: list[str],
    start:   str,
    end:     str,
    cash:    float,
    plot:    bool = False,
) -> dict:
    """
    Execute the Backtrader backtest and return a results dict.

    Returns
    -------
    dict with keys:
        start_value, end_value, total_return_pct,
        sharpe, max_drawdown_pct, trade_log, trades_df
    """

    # ── 1. Download data (with 1-year warm-up prepended) ─────────────────
    # The strategy needs ~252 bars of history before it can compute IV rank
    # and momentum signals.  We download from (start - 400 calendar days) so
    # the very first Monday of the official backtest period is signal-ready.
    from datetime import datetime, timedelta
    warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=400)
                    ).strftime("%Y-%m-%d")

    console.print(f"\n[bold cyan]Downloading data for {len(tickers)} tickers "
                  f"(warm-up from {warmup_start}, backtest {start} → {end})…"
                  f"[/bold cyan]")
    feeds = load_feeds(tickers, start=warmup_start, end=end, min_bars=60)
    if not feeds:
        console.print("[red]No data could be loaded — aborting.[/red]")
        return {}
    console.print(f"[green]✓ Loaded {len(feeds)} / {len(tickers)} tickers[/green]\n")

    # ── 2. Build cerebro ─────────────────────────────────────────────────
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(cash)

    # Commission scheme (applied to any BT orders; synthetic options manage
    # their own commission directly in the strategy via broker.add_cash)
    cerebro.broker.addcommissioninfo(OptionsCommission())

    for ticker, feed in feeds.items():
        cerebro.adddata(feed, name=ticker)

    cerebro.addstrategy(LongGammaStrategy)

    # Built-in analysers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        _name="sharpe",
                        riskfreerate=0.053,
                        annualize=True,
                        timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,     _name="returns")

    # ── 3. Run ───────────────────────────────────────────────────────────
    console.print("[bold cyan]Running backtest…[/bold cyan]")
    start_value = cerebro.broker.getvalue()
    results     = cerebro.run()
    end_value   = cerebro.broker.getvalue()
    strat       = results[0]

    # ── 4. Extract analytics ─────────────────────────────────────────────
    sharpe_val = (strat.analyzers.sharpe.get_analysis()
                  .get("sharperatio", None))
    dd_info    = strat.analyzers.drawdown.get_analysis()
    max_dd     = dd_info.get("max", {}).get("drawdown", 0.0)
    trade_info = strat.analyzers.trades.get_analysis()

    total_return = (end_value / start_value - 1) * 100

    # ── 5. Build trade DataFrame ─────────────────────────────────────────
    trades_df = pd.DataFrame(strat.trade_log) if strat.trade_log else pd.DataFrame()

    return {
        "start_value":       start_value,
        "end_value":         end_value,
        "total_return_pct":  round(total_return, 2),
        "sharpe":            round(sharpe_val, 3) if sharpe_val else None,
        "max_drawdown_pct":  round(max_dd, 2),
        "trade_info":        trade_info,
        "trade_log":         strat.trade_log,
        "trades_df":         trades_df,
        "cerebro":           cerebro,
    }


# ── Rich report ───────────────────────────────────────────────────────────────

def print_report(results: dict, start: str, end: str):
    if not results:
        return

    df  = results.get("trades_df", pd.DataFrame())
    ret = results["total_return_pct"]
    ret_color = "green" if ret >= 0 else "red"

    # ── Summary panel ────────────────────────────────────────────────────
    console.rule("[bold]Backtest Summary")
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column("Metric", style="dim")
    summary.add_column("Value",  justify="right")

    summary.add_row("Period",         f"{start}  →  {end}")
    summary.add_row("Start value",    f"${results['start_value']:,.0f}")
    summary.add_row("End value",      f"${results['end_value']:,.0f}")
    summary.add_row("Total return",
                    f"[{ret_color}]{ret:+.2f}%[/{ret_color}]")
    summary.add_row("Sharpe ratio",
                    str(results["sharpe"]) if results["sharpe"] else "n/a")
    summary.add_row("Max drawdown",   f"{results['max_drawdown_pct']:.2f}%")

    if not df.empty:
        n_trades  = len(df)
        n_wins    = int((df["pnl"] > 0).sum())
        win_rate  = n_wins / n_trades * 100 if n_trades else 0
        avg_win   = df.loc[df["pnl"] > 0, "pnl"].mean() if n_wins else 0
        avg_loss  = df.loc[df["pnl"] < 0, "pnl"].mean() if (df["pnl"] < 0).any() else 0
        total_pnl = df["pnl"].sum()

        summary.add_row("Total trades",  str(n_trades))
        summary.add_row("Win rate",      f"{win_rate:.1f}%  ({n_wins}/{n_trades})")
        summary.add_row("Avg win",       f"${avg_win:+.0f}")
        summary.add_row("Avg loss",      f"${avg_loss:+.0f}")
        summary.add_row("Total P&L",     f"${total_pnl:+,.0f}")

    console.print(summary)

    if df.empty:
        console.print("[yellow]No trades were completed.[/yellow]")
        return

    # ── Trade breakdown by type ──────────────────────────────────────────
    console.rule("[bold]By Trade Type")
    type_tbl = Table(box=box.SIMPLE)
    type_tbl.add_column("Type",        style="cyan")
    type_tbl.add_column("Direction",   style="magenta")
    type_tbl.add_column("Count",       justify="right")
    type_tbl.add_column("Win %",       justify="right")
    type_tbl.add_column("Total P&L",   justify="right")
    type_tbl.add_column("Avg P&L",     justify="right")

    grouped = df.groupby(["trade_type", "direction"])
    for (ttype, direction), grp in grouped:
        wins = int((grp["pnl"] > 0).sum())
        wr   = wins / len(grp) * 100
        tot  = grp["pnl"].sum()
        avg  = grp["pnl"].mean()
        color = "green" if tot >= 0 else "red"
        type_tbl.add_row(
            ttype, direction, str(len(grp)),
            f"{wr:.0f}%",
            f"[{color}]${tot:+,.0f}[/{color}]",
            f"[{color}]${avg:+,.0f}[/{color}]",
        )
    console.print(type_tbl)

    # ── Per-ticker breakdown ─────────────────────────────────────────────
    console.rule("[bold]By Ticker")
    tick_tbl = Table(box=box.SIMPLE)
    tick_tbl.add_column("Ticker",     style="cyan")
    tick_tbl.add_column("Trades",     justify="right")
    tick_tbl.add_column("Win %",      justify="right")
    tick_tbl.add_column("Total P&L",  justify="right")
    tick_tbl.add_column("Exit Reason",)

    for ticker, grp in df.groupby("ticker"):
        wins  = int((grp["pnl"] > 0).sum())
        wr    = wins / len(grp) * 100
        tot   = grp["pnl"].sum()
        color = "green" if tot >= 0 else "red"
        reasons = ", ".join(grp["reason"].value_counts().index[:3])
        tick_tbl.add_row(
            ticker, str(len(grp)), f"{wr:.0f}%",
            f"[{color}]${tot:+,.0f}[/{color}]",
            reasons,
        )
    console.print(tick_tbl)

    # ── Full trade log ───────────────────────────────────────────────────
    console.rule("[bold]Full Trade Log")
    log_tbl = Table(box=box.SIMPLE, show_lines=False)
    for col in ("open_date", "close_date", "ticker", "trade_type", "direction",
                "n_contracts", "premium", "pnl", "reason"):
        log_tbl.add_column(col, justify="right" if col in ("n_contracts","premium","pnl") else "left")

    df_sorted = df.sort_values("open_date")
    for _, row in df_sorted.iterrows():
        pnl   = float(row["pnl"])
        color = "green" if pnl >= 0 else "red"
        log_tbl.add_row(
            str(row["open_date"]),
            str(row["close_date"]),
            str(row["ticker"]),
            str(row["trade_type"]),
            str(row["direction"]),
            str(int(row["n_contracts"])),
            f"{float(row['premium']):.2f}",
            f"[{color}]{pnl:+.0f}[/{color}]",
            str(row["reason"]),
        )
    console.print(log_tbl)

    # ── Save CSV ─────────────────────────────────────────────────────────
    out = ROOT / "data" / "backtest_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"\n[dim]Trade log saved to {out}[/dim]")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Long-Gamma Backtrader backtest")
    parser.add_argument("--start",   default="2022-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    parser.add_argument("--cash",    type=float, default=100_000)
    parser.add_argument("--tickers", nargs="+",  default=None)
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    tickers = list(dict.fromkeys(args.tickers or WATCHLIST))   # deduplicate

    results = run_backtest(
        tickers=tickers,
        start=args.start,
        end=args.end,
        cash=args.cash,
        plot=args.plot,
    )

    print_report(results, args.start, args.end)

    if args.plot and results.get("cerebro"):
        results["cerebro"].plot(style="candlestick")


if __name__ == "__main__":
    main()
