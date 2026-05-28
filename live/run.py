"""
Live trading entry point.

Usage
-----
# Single Monday evaluation (enter positions):
python -m live.run --mode monday

# Daily position check at 15:45 ET:
python -m live.run --mode daily

# Show open positions:
python -m live.run --mode status

# Daemon — schedule both automatically (Monday 07:00 ET + daily 15:45 ET):
python -m live.run --daemon

# Dry run (signals + logs, no real orders):
python -m live.run --mode monday --dry-run
python -m live.run --daemon --dry-run

Credentials
-----------
Set environment variables before running (or create a .env file in the
project root — it is loaded automatically):

    ALPACA_API_KEY=<your_key>
    ALPACA_SECRET_KEY=<your_secret>
    ALPACA_PAPER=true          # always use paper=true until you're ready for live
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Load .env before importing anything that needs credentials ─────────────────
try:
    from dotenv import load_dotenv
    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[live/run] Loaded credentials from {env_file}")
    else:
        print(f"[live/run] No .env file at {env_file} — relying on environment variables")
except ImportError:
    print("[live/run] python-dotenv not installed; skipping .env load. "
          "Run: pip install python-dotenv")

import schedule
from rich.console import Console
from rich.table   import Table
from rich         import box

from live.alpaca_client   import AlpacaClient, AlpacaClientError
from live.event_trader    import EventTrader
from live.position_manager import PositionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log     = logging.getLogger(__name__)
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_client() -> AlpacaClient:
    try:
        return AlpacaClient()
    except AlpacaClientError as exc:
        console.print(f"[red bold]Credential error:[/red bold] {exc}")
        sys.exit(1)


def run_monday(dry_run: bool = False):
    console.print("\n[bold cyan]── Monday Evaluation ──[/bold cyan]")
    client = _make_client()
    trader = EventTrader(client)
    trader.run_monday_evaluation(dry_run=dry_run)


def run_daily(dry_run: bool = False):
    console.print("\n[bold cyan]── Daily Position Check ──[/bold cyan]")
    client  = _make_client()
    manager = PositionManager(client)
    manager.run(dry_run=dry_run)


def run_status():
    console.print("\n[bold cyan]── Open Positions ──[/bold cyan]")
    client  = _make_client()
    manager = PositionManager(client)

    # Show Alpaca account snapshot
    try:
        nav = client.account_value()
        console.print(f"Account NAV: [green]${nav:,.0f}[/green]\n")
    except Exception as exc:
        console.print(f"[yellow]Could not fetch account value: {exc}[/yellow]")

    manager.print_summary()

    # Show live Alpaca option positions
    positions = client.get_open_positions()
    option_pos = [p for p in positions
                  if hasattr(p, "asset_class") and str(p.asset_class) == "us_option"]
    if option_pos:
        console.rule("[bold]Live Alpaca Option Positions")
        tbl = Table(box=box.SIMPLE)
        tbl.add_column("Symbol")
        tbl.add_column("Qty",     justify="right")
        tbl.add_column("Avg Cost", justify="right")
        tbl.add_column("Market Value", justify="right")
        tbl.add_column("Unrealised P&L", justify="right")
        for p in option_pos:
            pnl   = float(p.unrealized_pl or 0)
            color = "green" if pnl >= 0 else "red"
            tbl.add_row(
                str(p.symbol),
                str(p.qty),
                f"${float(p.avg_entry_price or 0):.2f}",
                f"${float(p.market_value or 0):,.2f}",
                f"[{color}]${pnl:+,.2f}[/{color}]",
            )
        console.print(tbl)
    else:
        console.print("[dim]No option positions in Alpaca account[/dim]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Long-Gamma live trading (Alpaca paper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["monday", "daily", "status"],
        default=None,
        help="monday = enter new positions; daily = manage exits; status = show open",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (Monday 07:00 ET entry + daily 15:45 ET exit checks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals, log intent, but do NOT submit any orders",
    )
    args = parser.parse_args()

    if args.dry_run:
        console.print("[yellow bold]DRY RUN — no orders will be placed[/yellow bold]")

    if args.daemon:
        console.print("[bold green]Daemon mode starting…[/bold green]")
        console.print("  Monday 07:00 ET → new position evaluation")
        console.print("  Daily  15:45 ET → profit/stop/DTE exits\n")

        schedule.every().monday.at("07:00").do(run_monday,  dry_run=args.dry_run)
        schedule.every().day.at("15:45").do(run_daily,      dry_run=args.dry_run)

        while True:
            schedule.run_pending()
            time.sleep(30)

    elif args.mode == "monday":
        run_monday(dry_run=args.dry_run)

    elif args.mode == "daily":
        run_daily(dry_run=args.dry_run)

    elif args.mode == "status":
        run_status()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
